from rcn_tool_a import rcn_tool_a
import torch
from config import cfg


class rcn_tool_b(rcn_tool_a):
    def roi_pooling_cuda(self, features, proposal, label=None, stride=cfg.feature_stride, pool=None, batch=False):
        if batch == True:
            batch_output = []
            batch_label = []
            if label != None:
                batch_label.extend([j for i in label for j in i])
                batch_label = torch.stack(batch_label)
            outputs = pool(features, proposal)
            batch_output = outputs
            class_num = [0] * 6
            return batch_output, batch_label, class_num
        else:
            if len(features.size()) == 3:
                batch_size, num_channels, data_width = features.size()
                batch_output = []
                batch_label = []
                for index in range(batch_size):
                    data = features[index]
                    this_proposal = proposal[index]
                    outputs = pool(data, this_proposal)
                    batch_output.append(outputs)
                    if label != None:
                        batch_label.extend([i for i in label[index]])
                if label != None:
                    batch_label = torch.stack(batch_label)

                class_num = [0] * 5
                return batch_output, batch_label, class_num
            else:
                batch_output = []
                batch_label = []
                data = features
                this_proposal = proposal
                num_proposal = this_proposal.size()[0]
                outputs = pool(data, this_proposal)
                batch_output.extend([outputs[i, :] for i in range(num_proposal)])
                if label != None:
                    batch_label.extend(label)
                batch_output = torch.stack(batch_output, 0)
                return batch_output