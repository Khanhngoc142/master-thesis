from xml.etree import ElementTree as ET
from extractor.crohme_parser.inkml import Ink

if __name__ == "__main__":
    file_path = "/home/lap13639/Workplace/git/github/master-thesis/hmer/data/CROHME_full_v2/CROHME2013_data/TestINKML/118_em_232.inkml"
    # ink = Ink(file_path)
    tree = ET.parse(file_path)
