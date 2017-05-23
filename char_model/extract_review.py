import json
import codecs
from nltk.tokenize import sent_tokenize

def processReview(review):

    review_text = review["text"]
    review_text = review_text.strip().replace('\n', ' ').replace('\r', ' ')
    return review_text

def extract_review(review_ip_file, review_op_file):
    op_file = codecs.open(review_op_file,'wb',encoding='utf8')
    with open(review_ip_file) as ip_file:
        while True:
            lines = ip_file.readlines(1000)
            if not lines:
                break
            for line in lines:
                review = json.loads(line)
                review_text = processReview(review)
                op_file.write("<SOR> " + review_text + " <EOR>")
                op_file.write('\n')
    op_file.close()

extract_review('yelp_academic_dataset_review.json', 'yelp_review_data')
