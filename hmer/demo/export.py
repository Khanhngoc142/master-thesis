from extractor.crohme_parser.export import export_crohme_data

if __name__ == "__main__":
    limit = 30
    for dataset in [
        'train',
        'valid',
        # 'test'
    ]:
        print("EXPORT {} DATASET".format(dataset.upper()))
        export_crohme_data(datasets=dataset, limit=limit)
