import numpy as np
from utilities.fs import get_path
from utilities import plt_draw


def process_line(line, ground_truth=False):
    line = line.strip().split()
    img_path = line[0]
    if ground_truth:
        line = np.array(line[1:]).reshape((-1, 5))
        cls_ = line[:, 0]
        coords_ = line[:, 1:]
        conf_ = np.ones(cls_.shape, dtype=np.float)
    else:
        line = np.array(line[1:]).reshape((-1, 6))
        cls_ = line[:, 0]
        coords_ = line[:, 2:]
        conf_ = line[:, 1]
    return img_path, (cls_.astype('int'), conf_.astype('float'), coords_.astype('float'))


if __name__ == "__main__":
    # label_file = get_path("/home/lap13639/Workplace/git/github/master-thesis/hmer/training/data/CROHME_2013_valid/labels.txt")
    # with open(label_file, 'r') as f:
    #     data = f.readlines()
    #
    # data = dict([process_line(line, ground_truth=True) for line in data])
    # chosen_img = "training/data/CROHME_2013_valid/rit_42160_4{}.png"
    #
    # classes, conf, boxes = data[chosen_img.format('')]
    #
    # plt_draw.visualize_img_w_boxes(chosen_img.format(''), boxes, classes, conf, ncols=4)

    # data = process_line('training/data/CROHME_2013_valid/120_em_283.png 45 0.9987646341323853 240.30596923828125 145.2368927001953 259.0736999511719 167.2515411376953 45 0.9975433945655823 48.05730438232422 132.5530242919922 67.17404174804688 159.5345001220703 8 0.7871992588043213 160.09597778320312 172.75282287597656 171.71621704101562 182.01290893554688 36 0.6267387866973877 151.24269104003906 115.041748046875 183.8988800048828 138.51177978515625 58 0.5850775837898254 181.73876953125 171.0672149658203 204.1544189453125 187.77122497558594 13 0.5354081392288208 92.11639404296875 144.6113739013672 106.85706329345703 150.90447998046875 58 0.4563288688659668 112.46804809570312 161.969970703125 142.6663360595703 181.8038787841797 18 0.4256831109523773 147.7433319091797 179.88290405273438 148.6345977783203 185.14794921875 18 0.21992740035057068 150.32171630859375 177.9642333984375 151.78536987304688 186.1099853515625 18 0.1721428483724594 70.5860595703125 152.15261840820312 71.40522766113281 160.37998962402344 19 0.16850027441978455 190.23655700683594 131.06687927246094 198.21981811523438 141.15159606933594 18 0.16037000715732574 73.56039428710938 153.0880584716797 74.20402526855469 159.94956970214844 19 0.11978396773338318 212.00181579589844 182.02627563476562 220.17921447753906 189.02833557128906',
    #                     ground_truth=False)
    data = process_line('training/data/CROHME_2013_test/120_em_292.png 69 0.9998735189437866 154.2259063720703 136.94482421875 176.4447784423828 184.4025421142578 69 0.8484979271888733 76.90184020996094 133.935302734375 95.38138580322266 181.2236785888672 1 0.8126317858695984 48.111873626708984 128.99856567382812 57.35300064086914 167.46273803710938 1 0.8119179606437683 188.93771362304688 126.50880432128906 197.8476104736328 140.04830932617188 45 0.740942656993866 203.24667358398438 130.66099548339844 212.12249755859375 139.64515686035156 9 0.615315318107605 113.33219146728516 156.14601135253906 128.6387939453125 158.04248046875 2 0.5556000471115112 241.2340545654297 130.28443908691406 259.4927062988281 179.61062622070312 12 0.22773809731006622 221.69398498535156 118.66785430908203 231.26564025878906 138.462890625'
                        .replace('_test', '_valid'),
                        ground_truth=False)

    classes, conf, boxes = data[1]

    plt_draw.visualize_img_w_boxes(data[0], boxes, classes, conf, ncols=4)
