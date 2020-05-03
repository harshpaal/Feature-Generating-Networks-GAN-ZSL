
#Arguments--------------------------------------------

dataset = 'awa2'
gzsl = False
latent_dim = 128
n_critic = 5
lmbda = 10
beta = 0.01
batch_size = 128
n_epochs = 15
use_cls_loss = False
visualize = True
#---------------------------------------------------------

import torch
from torch.utils.data import DataLoader


if dataset == 'awa2' or dataset == 'awa1':
    x_dim = 2048
    attr_dim = 85
    n_train = 40
    n_test = 10
elif dataset == 'cub':
    x_dim = 2048
    attr_dim = 312
    n_train = 150
    n_test = 50
elif dataset == 'sun':
    x_dim = 2048
    attr_dim = 102
    n_train = 645
    n_test = 72
else:
    raise NotImplementedError

n_epochs = n_epochs

# trainer object for mini batch training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_agent = Trainer(
    device, x_dim, latent_dim, attr_dim,
    n_train=n_train, n_test=n_test, gzsl=gzsl,
    n_critic=n_critic, lmbda=lmbda, beta=beta,
    batch_size=batch_size
)

params = {
    'batch_size': batch_size,
    'shuffle': True,
    'num_workers': 0,
    'drop_last': True
}

train_dataset = ZSLDataset(dataset, n_train, n_test, gzsl)
train_generator = DataLoader(train_dataset, **params)

# =============================================================
# PRETRAIN THE SOFTMAX CLASSIFIER
# =============================================================
model_name = "%s_disc_classifier" % dataset
#success = train_agent.load_model(model=model_name)
success = False
if success:
    print("Discriminative classifier parameters loaded...")
else:
    print("Training the discriminative classifier...")
    for ep in range(1, n_epochs + 1):
        loss = 0
        for idx, (img_features, label_attr, label_idx) in enumerate(train_generator):
            l = train_agent.fit_classifier(img_features, label_attr, label_idx)
            loss += l

        print("Loss for epoch: %3d - %.4f" %(ep, loss))

    train_agent.save_model(model=model_name)

# =============================================================
# TRAIN THE GANs
# =============================================================
model_name = "%s_gan_generator" % dataset
#success = train_agent.load_model(model=model_name)
success = False
if success:
    print("\nGAN parameters loaded....")
else:
    print("\nTraining the GANS...")
    for ep in range(1, n_epochs + 1):
        loss_dis = 0
        loss_gan = 0
        for idx, (img_features, label_attr, label_idx) in enumerate(train_generator):
            l_d, l_g = train_agent.fit_GAN(img_features, label_attr, label_idx, use_cls_loss)
            loss_dis += l_d
            loss_gan += l_g

        print("Loss for epoch: %3d - D: %.4f | G: %.4f"\
                %(ep, loss_dis, loss_gan))

    train_agent.save_model(model=model_name)

# =============================================================
# TRAIN FINAL CLASSIFIER ON SYNTHETIC DATASET
# =============================================================

# create new synthetic dataset using trained Generator
seen_dataset = None
if gzsl:
    seen_dataset = train_dataset.gzsl_dataset

syn_dataset = train_agent.create_syn_dataset(
        train_dataset.test_classmap, train_dataset.attributes, seen_dataset)
final_dataset = ZSLDataset(dataset, n_train, n_test,
        gzsl=gzsl, train=True, synthetic=True, syn_dataset=syn_dataset)
final_train_generator = DataLoader(final_dataset, **params)

model_name = "%s_final_classifier" % dataset
#success = train_agent.load_model(model=model_name)
success = False
if success:
    print("\nFinal classifier parameters loaded....")
else:
    print("\nTraining the final classifier on the synthetic dataset...")
    for ep in range(1, n_epochs + 1):
        syn_loss = 0
        for idx, (img, label_attr, label_idx) in enumerate(final_train_generator):
            l = train_agent.fit_final_classifier(img, label_attr, label_idx)
            syn_loss += l

        # print losses on real and synthetic datasets
        print("Loss for epoch: %3d - %.4f" %(ep, syn_loss))

    train_agent.save_model(model=model_name)

# =============================================================
# TESTING PHASE
# =============================================================
test_dataset = ZSLDataset(dataset, n_train, n_test, gzsl=gzsl, train=False)
test_generator = DataLoader(test_dataset, **params)

print("\nFinal Accuracy on ZSL Task: %.3f" % train_agent.test(test_generator))
