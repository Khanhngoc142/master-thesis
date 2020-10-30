import os
from extractor.crohme_parser.export import export_crohme_data, find_weird_boxes, generate_extra_training_data, export_crohme_latex
from utilities.fs import get_source_root, load_object

if __name__ == "__main__":
    limit = 10
    reports = []

    root_path = get_source_root()
    # demo_lib = load_object(os.path.join(root_path, "demo-outputs", "lib.pkl"))
    # generate_extra_training_data(demo_lib, geo_aug=True, n_loop=5000, output_dir='training/extra_data')

    for dataset in [
        'train',
        'valid',
        # 'test'
    ]:
        print("EXPORT {} DATASET".format(dataset.upper()))
        # export_crohme_data(datasets=dataset, limit=limit, output_dir="training/data_aug")
        # export_crohme_data(datasets=dataset, limit=limit, output_dir="training/data", treo_aug=False)
        export_crohme_latex(datasets=dataset, limit=limit, output_dir="training/data")
    #     report = "\n".join(find_weird_boxes(datasets=dataset, limit=limit))
    #     reports.append(report)
    # reports = "\n".join(reports)
    # with open(os.path.join(get_source_root(), "demo-outputs/weird.txt"), 'w') as fout:
    #     fout.write(reports)
