from annotation_merge_utils import get_intersection_size


def _span_set_coverage(annotations_1, annotations_2):
    _sum = 0
    for i in range(len(annotations_1)):
        for j in range(len(annotations_2)):
            inters_size = get_intersection_size(annotations_1[i]['start_index'], annotations_1[i]['length'],
                                                annotations_2[j]['start_index'], annotations_2[j]['length'])

            _sum += inters_size / annotations_1[i]['length']

    return _sum


def _get_f1_for_pair(doc_info_h, doc_info_r):
    precision = _span_set_coverage(doc_info_r, doc_info_h) / len(doc_info_h)
    recall = _span_set_coverage(doc_info_h, doc_info_r) / len(doc_info_r)
    if precision + recall != 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0.0


def _get_f1_for_doc(document_annotations):
    results = [[0] * len(document_annotations) for _ in range(len(document_annotations))]
    f1s = []
    for h in range(len(document_annotations)):
        for r in range(h, len(document_annotations)):
            results[h][r] = _get_f1_for_pair(document_annotations[h], document_annotations[r])
            f1s.append(results[h][r])

    return sum(f1s) / len(f1s)


def _transform_data(document_data, clss):
    document_list = []
    for doc_data in document_data:
        temp = []
        for annotation_data in doc_data:
            if annotation_data['decision'] != clss:
                continue
            elif len(annotation_data['spans']) == 0:
                continue

            temp.append(annotation_data['spans'])

        if len(temp) < 2:
            continue
        else:
            document_list.append(temp)

    return document_list


def soft_f1(document_data, clss):
    document_list = _transform_data(document_data, clss)

    results = []
    for document_annotations in document_list:
        results.append(_get_f1_for_doc(document_annotations))

    return sum(results) / len(results)
