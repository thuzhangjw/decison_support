from cnn_model import cnn_model
from data_helper import train_data_generation, fold_generation
from config_helper import train_config_parser
from util import *


def train_routine(args):
    model_name = "RelationCNN" + args.model_name_suffix
    (seq_len, data_blocks, data_dicts,
     data_dict_sizes, word_vector) = train_data_generation(args)

    print("data blocks loaded")
    data_blocks_folded = fold_generation(data_blocks, args.num_fold)
    acc_list, f1_list = [], []
    print(len(data_blocks_folded))
    from random import shuffle
    shuffle(data_blocks_folded)
    for f, (train_blocks, test_blocks) in enumerate(data_blocks_folded):
        for block in train_blocks:
            print(block.shape)
        for block in test_blocks:
            print(block.shape)
        model = cnn_model(model_name=model_name,
                          seq_len=seq_len,  # length of largest sent
                          data_dict_sizes=data_dict_sizes,
                          data_dicts=data_dicts,
                          word_vector=word_vector,
                          args=args)
        acc, f1_score = model.cnn_train(train_blocks, test_blocks, f)

        print("Accuracy = %f" % acc)
        print("F1 score = %f" % f1_score)
        acc_list.append(acc)
        f1_list.append(f1_score)

    print("Average accuracy = ", np.mean(acc_list))
    print("Average F1 score = ", np.mean(f1_list))



if __name__ == "__main__":
    train_routine(train_config_parser.parse_args())
