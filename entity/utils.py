import numpy as np
import json
import logging

logger = logging.getLogger('root')

def batchify(samples, batch_size):
    """
    Batchfy samples with a batch size
    """
    num_samples = len(samples)

    list_samples_batches = []

    # if a sentence is too long, make itself a batch to avoid GPU OOM
    to_single_batch = []
    for i in range(0, len(samples)):
        if len(samples[i]['tokens']) > 350:
            to_single_batch.append(i)

    for i in to_single_batch:
        logger.info('Single batch sample: %s-%d', samples[i]['doc_key'], samples[i]['sentence_ix'])
        list_samples_batches.append([samples[i]])
    samples = [sample for i, sample in enumerate(samples) if i not in to_single_batch]

    for i in range(0, len(samples), batch_size):
        list_samples_batches.append(samples[i:i+batch_size])

    assert(sum([len(batch) for batch in list_samples_batches]) == num_samples)

    return list_samples_batches


def overlap(s1, s2):
    if s2.start_sent >= s1.start_sent and s2.start_sent <= s1.end_sent:
        return True
    if s2.end_sent >= s1.start_sent and s2.end_sent <= s1.end_sent:
        return True
    return False


def convert_dataset_to_samples(dataset, max_span_length, ner_label2id=None, training=False):
    """
    Extract sentences and gold entities from a dataset
    """
    data = [json.loads(line) for line in open(dataset)]
    # print(len(data))
    num_ner = 0
    total_spans = 0
    max_len = 0
    max_ner = 0
    samples = []

    for i, sent in enumerate(data[0][:]):
        num_ner += len(sent['entities'])
        sample = {}
        sample['tokens'] =  sent['tokens'] 
        sample['sent_length'] = len(sent['tokens'])
        sample['num_spans_o'] = 0

        sent_start = 0
        sent_end = len(sample['tokens'])

        max_len = len(sent['tokens'])
        max_ner = len(sent['entities'])

        sample['sent_start'] = sent_start
        sample['sent_end'] = sent_end

        sent_ner = {}
        for ner in sent['entities']:
            sent_ner[(ner[0], ner[1])] = ner[-1]

        span2id = {}
        sample['spans'] = []
        sample['spans_label'] = []
        sample['spans_o'] = []
        sample['spans_label_o'] = []

        for i in range(len(sent['tokens'])):
            for j in range(i, min(len(sent['tokens']), i+max_span_length)):
                sample['spans'].append((i+sent_start, j+sent_start, j-i+1))
                span2id[(i, j)] = len(sample['spans'])-1
                if (i, j) not in sent_ner:
                    sample['spans_label'].append(0)
                else:
                    sample['spans_label'].append(ner_label2id[sent_ner[(i, j)]])

        total_spans += len(sample['spans'])

        if len(sample['spans_label']) > 0:
            samples.append(sample)
            
    avg_length = sum([len(sample['tokens']) for sample in samples]) / len(samples)
    max_length = max([len(sample['tokens']) for sample in samples])
    logger.info('Extracted %d samples, with %d NER labels, %.3f avg input length, %d max length'%(len(samples), num_ner, avg_length, max_length))
    logger.info('Max Length: %d, max NER: %d'%(max_len, max_ner))
    logger.info('Num spans: %d'%(total_spans))
    return samples, num_ner

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_train_fold(data, fold):
    print('Getting train fold %d...'%fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold+1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i < l or i >= r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d'%(len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data

def get_test_fold(data, fold):
    print('Getting test fold %d...'%fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold+1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i >= l and i < r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d'%(len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data

