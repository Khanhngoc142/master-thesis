import os
import argparse
from extractor.crohme_parser.export import export_crohme_data, find_weird_boxes
from utilities.fs import get_source_root

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export CROHME data.')
    parser.add_argument('dataset', type=str)
    parser.add_argument('--save_folder', type=str, default='training/data')
    parser.add_argument('--geo_aug', type=bool, default=False)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    # reports = []
    for dataset in args.dataset.split('+'):
        print("EXPORT {} DATASET".format(dataset.upper()))
        # export_crohme_data(datasets=dataset, limit=limit, output_dir="training/data_aug")
        export_crohme_data(datasets=dataset, limit=args.limit, output_dir=args.save_folder, treo_aug=args.geo_aug)
    #     report = "\n".join(find_weird_boxes(datasets=dataset, limit=limit))
    #     reports.append(report)
    # reports = "\n".join(reports)
    # with open(os.path.join(get_source_root(), "demo-outputs/weird.txt"), 'w') as fout:
    #     fout.write(reports)
    print("DONE")
