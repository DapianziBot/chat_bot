from config import *
from tqdm import tqdm
import pickle
import json
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import pysparnn.cluster_index as ci
import logging

jieba.setLogLevel(logging.INFO)


def prepare_baike():
    qa_dict = {}
    count = 0
    f_baike = open(QA_BAIKE_CORPUS, 'r', encoding='utf-8')
    for line in tqdm(f_baike.readlines(), ascii=True, total=200000):
        if count >= 200000:
            break
        qa = json.loads(line)
        qa_dict[qa['title']] = {
            'answer': qa['answer'].replace("\r\n", ""),
            'entity': qa['category'].split('-')[0],
            'cutted': jieba.lcut(qa['title'])
        }
        count += 1

    json.dump(qa_dict, open(QA_BAIKE_DICT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


#
def build_qa_vectors():
    qa_dict = json.load(open(QA_BAIKE_DICT, 'r', encoding='utf-8'))
    # 分词完成的问题
    lines_cuted = [' '.join(q['cutted']) for q in qa_dict.values()]
    # 初始化向量化的方法
    tv = TfidfVectorizer()
    # 训练
    tv.fit(lines_cuted)
    # 得到特征向量
    features_vec = tv.transform(lines_cuted)
    # 构造索引
    cp = ci.MultiClusterIndex(features_vec, qa_dict.keys())
    pickle.dump(tv, open(QA_BAIKE_TFIDF_VECTORS, 'wb'))
    pickle.dump(cp, open(QA_BAIKE_CLUSTER_INDEX, 'wb'))


def search_distance(data, k=1, k_clusters=10, return_distance=True):
    cp = pickle.load(open(QA_BAIKE_CLUSTER_INDEX, 'rb'))
    tv = pickle.load(open(QA_BAIKE_TFIDF_VECTORS, 'rb'))
    vec = tv.transform(data)
    distance = cp.search(vec, k=k, k_clusters=k_clusters, return_distance=return_distance)
    return distance


if __name__ == '__main__':
    # prepare_baike()
    build_qa_vectors()
    data = [
        '衬衫沾染茶渍如何清洗?',
        '谁能告诉我一代精灵和二代精灵有什么区别吗?',
        '请介绍ibm公司的概况',
        '解放战争后期撤到台湾的国军有多少？'
    ]
    distance = search_distance([' '.join(jieba.lcut(x)) for x in data], k=2)
    print(distance)
