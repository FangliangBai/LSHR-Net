import sys

sys.path.insert(0, './')
from solver import *

flags = tf.app.flags
conf = flags.FLAGS
# ==================================================== Data Source =====================================================
# Data - MNIST
# flags.DEFINE_string("train_hr_list",        "data/hrImg_32X32_1001-2000.txt", "")
# flags.DEFINE_string("train_lr_list",        "data/lrImg_simula_16X16_1001-2000.txt", "")
# flags.DEFINE_string("val_hr_list",          "data/valid_hrImgPath_Simulation.txt", "")
# flags.DEFINE_string("val_lr_list",          "data/valid_lrImgPath_Simulation_X2.txt", "")
# flags.DEFINE_string("test_lr_list",         "data/test_lr_list.txt", "")
# Data - DIV2K
flags.DEFINE_string("train_hr_list",        "/home/kent/Downloads/DIV2K/DIV2K_train_HR/", "")
flags.DEFINE_string("train_lr_list",        "/home/kent/Downloads/DIV2K/DIV2K_train_LR/", "")
flags.DEFINE_string("val_hr_list",          "/home/kent/Downloads/DIV2K/valid images/", "")
flags.DEFINE_string("val_lr_list",          "/home/kent/Downloads/DIV2K/test images/", "")
flags.DEFINE_string("test_lr_list",         "data/test_lr_list.txt", "")
# Data - cifar10
flags.DEFINE_string("cifar10_path",         "/media/kent/DISK2/data/", "")

# ======================================================= Result =======================================================
# Results save path
flags.DEFINE_string("model_dir",            "/media/kent/DISK2/sr_spc/models", "trained model save path")
flags.DEFINE_string("validate_dir",         "/media/kent/DISK2/sr_spc/validate", "validated images save path")
flags.DEFINE_string("test_dir",             "/media/kent/DISK2/sr_spc/test", "sampled images save path")

# ===================================================== Parameters =====================================================
# GPU params
flags.DEFINE_boolean("use_gpu",             True, "whether to use gpu for training")
flags.DEFINE_integer("device_id",           0, "gpu device id")

# Data params
flags.DEFINE_string("dataset",              "DIV2K", "Options: [DIV2K, cifar10, MNIST]")
flags.DEFINE_integer("num_train_images",    11, "total number of images")
flags.DEFINE_integer("num_val_images",      11, "total number of images for training")
flags.DEFINE_integer("im_size",             256, "for DIV2K: the size of raw image")
flags.DEFINE_integer("hr_size",             256, "hr image size")
flags.DEFINE_integer("lr_size",             128, "lr image size")
flags.DEFINE_integer("img_channel",         1, "img_channel, use 1 for DIV2K")

# Training params
flags.DEFINE_boolean("train_only",          True, "whether to train without test")
flags.DEFINE_boolean("test_only",           False, "whether to test only")
flags.DEFINE_integer("num_epoch",           120000, "train epoch num")
flags.DEFINE_integer("batch_size",          11, "batch_size")
flags.DEFINE_integer("shuffle_size",        10000, "the random shuffle size of dataset")

# Learning params
flags.DEFINE_float("learning_rate",         1.0e-4, "initial learning rate")
flags.DEFINE_float("decay_steps",           10000, "decay_steps for learning rate")
flags.DEFINE_float("decay_rate",            0.5, "decay rate for learning rate")
flags.DEFINE_boolean("charbonnier_loss",    True, "whether to train without test")

# Model params
flags.DEFINE_integer("num_measure",         41, "number of measure")
flags.DEFINE_string("mask_type",            "Bernoulli", "Options: [trained, Bernoulli, trained_binary, dropout]")
flags.DEFINE_float("dropout",               0.5, "keep probability for dropout mask")
flags.DEFINE_boolean("linear_mapping_only", False, "whether to train linear mapping only")
flags.DEFINE_boolean("restore_linear_part", False, "whether to restore linear model if restore_linear_part is true")
flags.DEFINE_boolean("train_residual_only", False, "whether to train residual block only with pretrained linear part")
flags.DEFINE_boolean("restore_whole_model", False, "whether to restore whole model")

flags.DEFINE_string("linear_model",     "/media/kent/DISK2/sr_spc/experiment_54/models/model.ckpt-2800000","")
flags.DEFINE_string("whole_model",      "/media/kent/DISK2/sr_spc/experiment_61/models/model.ckpt-12000","")


def main(_):
    solver = Solver()

    if conf.train_only:
        solver.train()
    elif conf.test_only:
        solver.test()


if __name__ == '__main__':
    tf.app.run()
