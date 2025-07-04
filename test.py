import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser
import pdb
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F
class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out
class MLP(nn.Module):
    def __init__(self, n_i, n_h, n_o):
        super(MLP, self).__init__()

        self.linear1 = NormedLinear(n_i, n_h)
        self.linear2 = NormedLinear(n_h, n_o)

        self.ac = nn.ReLU()
    def forward(self, input):

        return self.linear2(self.ac((self.linear1(input))))
# net = MLP(100,50,3).to("cuda")
# loss = nn.CrossEntropyLoss()
# sm = nn.Softmax(dim=-1)
# optimizer = optim.SGD(net.parameters(), lr=0.05)
def main(config):
    logger = config.get_logger('test')
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=256,
        shuffle=False,
        training=False,
        num_workers=12
    )

    # build model architecture
    if 'returns_feat' in config['arch']['args']:
        model = config.init_obj('arch', module_arch, allow_override=True, returns_feat=True)
    else:
        model = config.init_obj('arch', module_arch) 

    # get function handles of loss and metrics 
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict, strict=False)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
 
    total_metrics = torch.zeros(len(metric_fns))

    num_classes = config._config["arch"]["args"]["num_classes"]
    confusion_matrix = torch.zeros(num_classes, num_classes).cuda()


    get_class_acc = True
    if get_class_acc:
        train_data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size=256,
            training=True
        ) 
        train_cls_num_list = np.array(train_data_loader.cls_num_list)
        # print(train_cls_num_list)
        many_shot = train_cls_num_list > 100
        medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
        few_shot = train_cls_num_list < 20



    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            output = (output['logits'].transpose(0, 1)[0] + output['logits'].transpose(0, 1)[1] +
                      output['logits'].transpose(0, 1)[2])/ 3

            batch_size = data.shape[0]

            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size


            for t, p in zip(target.view(-1), output.argmax(dim=1).view(-1)):
                confusion_matrix[t.long(), p.long()] += 1


    acc_per_class = confusion_matrix.diag()/confusion_matrix.sum(1) 
    acc = acc_per_class.cpu().numpy() 

    many_shot_acc = acc[many_shot].mean()
    medium_shot_acc = acc[medium_shot].mean()
    few_shot_acc = acc[few_shot].mean() 

    n_samples = len(data_loader.sampler)
    log = {}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })

    if get_class_acc:
        log.update({
            "many_class_num": many_shot.sum(),
            "medium_class_num": medium_shot.sum(),
            "few_class_num": few_shot.sum(),
            "many_shot_acc": many_shot_acc,
            "medium_shot_acc": medium_shot_acc,
            "few_shot_acc": few_shot_acc,
        })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
