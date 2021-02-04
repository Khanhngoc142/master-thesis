from utilities import BB_To_Tree
import timeout_decorator


# @timeout_decorator.timeout(180, exception_message='Time out', timeout_exception=TimeoutError)
def latex_generator(line, is_pred=True):
    generator = BB_To_Tree.BBParser()
    return generator.process(line, True)


if __name__ == "__main__":
    # generator = BB_To_Tree.BBParser()
    predicted_file = '/home/ubuntu/data/ssd_final_result/aug_geo_multithreshnmstest.txt'

    with open(predicted_file) as f:
        lines = f.readlines()
        # latex_result = [line.split(' ')[0].replace('.png', '') + '\t' + latex_generator(line, True) for line in lines]
        latex_result = []
        err_result = []
        for line in lines:
            file_name = line.split(' ')[0].replace('.png', '')
            try:
                latex_result.append(file_name + '\t' + latex_generator(line, True))
            except TimeoutError:
                err_result.append(file_name)

    with open('/home/ubuntu/workspace/mine/master-thesis.git/hmer/metadata/latex_files/predicted_latex_aug_geo_notm_add_rules.txt',
              'w') as wf:
        wf.write('\n'.join(latex_result))

    with open('/home/ubuntu/workspace/mine/master-thesis.git/hmer/metadata/latex_files/timeout_files_aug_geo_notm_add_rules.txt',
              'w') as wf:
        wf.write('\n'.join(err_result))
