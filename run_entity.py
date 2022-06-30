import json
import argparse
import os
import sys
import random
import logging
import time
from tqdm import tqdm
import numpy as np
import json
from shared.data_structures import Dataset
from shared.const import task_ner_labels, get_labelmap
from entity.utils import convert_dataset_to_samples, batchify, NpEncoder
from entity.models import EntityModel

from transformers import AdamW, get_linear_schedule_with_warmup
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')

def save_model(model, args):
    """
    Save the model to the output directory
    """
    logger.info('Saving model to %s...'%(args.output_dir))
    torch.save(model.bert_model.state_dict(), args.output_dir+str('/best_model.m'))

def output_ner_predictions(model, batches, dataset, output_file):
    """
    Save the prediction as a json file
    """
    ner_result = {}
    start_result = {}
    end_result = {}
    span_hiddens = []
    hard_neg_hiddens = []
    tot_pred_ett = 0
    k = 0
    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        pred_hidden = output_dict['ner_last_hidden']
        pred_probs = output_dict['ner_probs']
        span_hidden = []
        hard_neg_hidden = []
        for sample, preds, hiddens, probs in zip(batches[i], pred_ner, pred_hidden, pred_probs):
            off = 0
            ner_result[str(k)] = []
            span_hid = []
            hard_neg_hid = []
            for span, pred, hidden, prob in zip(sample['spans'], preds, hiddens, probs):
                if pred == 0:
                    if max(prob) < 0.95:
                        hard_neg_hid.append(hidden)
                    continue
                ner_result[str(k)].append([span[0]+off, span[1]+off, ner_id2label[pred]])
                span_hid.append(hidden)
            span_hidden.append(span_hid)
            hard_neg_hidden.append(hard_neg_hid)
            k += 1
        span_hiddens.extend(span_hidden)
        hard_neg_hiddens.extend(hard_neg_hidden)

    print(len(ner_result))
    print(ner_result[str(1)])
    tot_pred_ett = len(list(ner_result.values()))
    with open('bb_ner_result.json', 'w') as f:
        json.dump(ner_result, f)
    with open('hidden.npy', 'wb') as f:
        np.save(f, span_hiddens)    
    with open('hard_neg_hidden.npy', 'wb') as f:
        np.save(f, hard_neg_hiddens)   
    logger.info('Total pred entities: %d'%tot_pred_ett)


def evaluate(model, batches, tot_gold):
    """
    Evaluate the entity model
    """
    logger.info('Evaluating...')
    cor = 0
    tot_pred = 0
    l_cor = 0
    l_tot = 0
    wrong_span = 0

    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            for gold, pred in zip(sample['spans_label'], preds):
                l_tot += 1
                if pred == gold:
                    l_cor += 1
                if pred != 0 and gold != 0 and pred == gold:
                    cor += 1
                if pred != 0:
                    tot_pred += 1
                    if gold == 0:
                        wrong_span += 1
                   
    acc = l_cor / l_tot
    logger.info('wrong spans count: %d'%(wrong_span))
    logger.info('Accuracy: %5f'%acc)
    logger.info('Cor: %d, Pred TOT: %d, Gold TOT: %d'%(cor, tot_pred, tot_gold))
    p = cor / tot_pred if cor > 0 else 0.0
    r = cor / tot_gold if cor > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    logger.info('P: %.5f, R: %.5f, F1: %.5f'%(p, r, f1))

    return f1

def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default=None, required=True, choices=['conll03', 'ontonotes', 'bc5cdr',  'twitter', 'wiki',])

    parser.add_argument('--data_dir', type=str, default=None, required=True, 
                        help="path to the preprocessed dataset")               
    parser.add_argument('--output_dir', type=str, default='entity_output', 
                        help="output directory of the entity model")

    parser.add_argument('--max_span_length', type=int, default=8, 
                        help="spans w/ length up to max_span_length are considered as candidates")
    parser.add_argument('--train_batch_size', type=int, default=16, 
                        help="batch size during training")
    parser.add_argument('--eval_batch_size', type=int, default=32, 
                        help="batch size during inference")
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                        help="learning rate for the BERT encoder")
    parser.add_argument('--task_learning_rate', type=float, default=5e-4, 
                        help="learning rate for task-specific parameters, i.e., classification head")
    parser.add_argument('--warmup_proportion', type=float, default=0, 
                        help="the ratio of the warmup steps to the total steps")
    parser.add_argument('--num_epoch', type=int, default=20, 
                        help="number of the training epochs")
    parser.add_argument('--print_loss_step', type=int, default=100, 
                        help="how often logging the loss value during training")
    parser.add_argument('--eval_per', type=float, default=.5, 
                        help="how often evaluating the trained model on dev set during training")
    parser.add_argument("--bertadam", action="store_true", help="If bertadam, then set correct_bias = False")
    parser.add_argument('--test_pred_filename', type=str, default="ent_pred_test.json", help="the prediction filename for the test set")

    parser.add_argument('--do_train', action='store_true', 
                        help="whether to run training")
    parser.add_argument('--train_shuffle', action='store_true',
                        help="whether to train with randomly shuffled data")
    parser.add_argument('--do_eval', action='store_true', 
                        help="whether to run evaluation")
    parser.add_argument('--eval_test', action='store_true', 
                        help="whether to evaluate on test set")

    parser.add_argument('--model', type=str, default='bert-base-uncased', 
                        help="the base model name (a huggingface model)")
    parser.add_argument('--bert_model_dir', type=str, default=None, 
                        help="the base model directory")
    parser.add_argument('--plm_hidden_size', type=int, default=768)                  
    parser.add_argument('--rate', type=float, default=0.95, help='samplinng rate for negatives')                  

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    args.train_data = os.path.join(args.data_dir, 'train-ds.json')
    args.dev_data = os.path.join(args.data_dir, 'dev.json')
    args.test_data = os.path.join(args.data_dir, 'test.json')
    setseed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))

    logger.info(sys.argv)
    logger.info(args)
    
    ner_label2id, ner_id2label = get_labelmap(task_ner_labels[args.task])
    
    num_ner_labels = len(task_ner_labels[args.task]) + 1
    model = EntityModel(args, num_ner_labels=num_ner_labels)

    print(ner_label2id)
    dev_samples, dev_ner = convert_dataset_to_samples(args.dev_data, args.max_span_length, ner_label2id=ner_label2id, training=False)
    dev_batches = batchify(dev_samples, args.eval_batch_size)

    if args.do_train:
        train_samples, train_ner = convert_dataset_to_samples(args.train_data, args.max_span_length, ner_label2id=ner_label2id, training=True)
        print(train_samples[:5])
        train_batches = batchify(train_samples, args.train_batch_size)
        best_result = 0.0

        param_optimizer = list(model.bert_model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                if 'PLM' in n],
            'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer
                if 'PLM' not in n], 'lr': args.task_learning_rate,
            'weight_decay': 0.0}
                ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=not(args.bertadam))
        t_total = len(train_batches) * args.num_epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total*args.warmup_proportion), t_total)
        
        tr_loss = 0
        tr_examples = 0
        global_step = 0
        eval_step = int(len(train_batches) * args.eval_per)
        for _ in tqdm(range(args.num_epoch)):
            if args.train_shuffle:
                random.shuffle(train_batches)
            for i in range(len(train_batches)):
                output_dict = model.run_batch(train_batches[i], training=True)
                loss = output_dict['ner_loss']
                loss.backward()

                tr_loss += loss.item()
                tr_examples += len(train_batches[i])
                global_step += 1

                optimizer.step()
                optimizer.zero_grad()

                if global_step % args.print_loss_step == 0:
                    logger.info('Epoch=%d, iter=%d, loss=%.5f'%(_, i, tr_loss / tr_examples))
                    tr_loss = 0
                    tr_examples = 0
                
                if global_step % eval_step == 0:
                    # print('Performance on train,,,,')
                    # f1 = evaluate(model, train_batches, train_ner)
                    # print('Performance on dev ... ')
                    f1 = evaluate(model, dev_batches, dev_ner)
                    if f1 > best_result:
                        best_result = f1
                        logger.info('!!! Best valid (epoch=%d): %.2f' % (_, f1*100))
                        save_model(model, args)


    if args.do_eval:
        model = EntityModel(args, num_ner_labels=num_ner_labels)
        model.bert_model.load_state_dict(torch.load(args.output_dir+str('/best_model.m')))
        logger.info('Best Dev Performance ...')
        evaluate(model, dev_batches, dev_ner)
        logger.info('Evaluating test...')
        test_samples, test_ner = convert_dataset_to_samples(args.test_data, args.max_span_length, ner_label2id=ner_label2id)
        test_batches = batchify(test_samples, args.eval_batch_size)
        evaluate(model, test_batches, test_ner)
        # prediction_file = os.path.join(args.output_dir, args.test_pred_filename)
        # output_ner_predictions(model, test_batches, args.test_data, output_file=prediction_file)


