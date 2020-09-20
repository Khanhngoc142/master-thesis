import os
from extractor.crohme_parser.export import export_crohme_data, find_weird_boxes
from utilities.fs import get_source_root

if __name__ == "__main__":
    limit = None
    reports = []
    for dataset in [
        'train',
        'valid',
        # 'test'
    ]:
        print("EXPORT {} DATASET".format(dataset.upper()))
        export_crohme_data(datasets=dataset, limit=limit, output_dir="training/data_aug")
    #     report = "\n".join(find_weird_boxes(datasets=dataset, limit=limit))
    #     reports.append(report)
    # reports = "\n".join(reports)
    # with open(os.path.join(get_source_root(), "demo-outputs/weird.txt"), 'w') as fout:
    #     fout.write(reports)
