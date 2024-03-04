from rcn_tool_b import rcn_tool_b


class rcn_tool_c(rcn_tool_b):
    def cal_loss2(self, cls_score, label):

        cross_entropy = 0
        keep = label != -1
        this_cls_score = cls_score[keep]
        this_label = label[keep]
        cross_entropy += self.loss(this_cls_score, this_label.cuda())
        cross_entropy = cross_entropy / len(keep)
        return cross_entropy

    def cal_loss(self, cls_score, box_pred, label, predict_offset, cls_weight, cor_weight):

        cross_entropy = 0
        
        keep = label != -1
        
        this_cls_score = cls_score[keep]
        this_label = label[keep]

        cross_entropy += self.loss(this_cls_score, this_label.cuda())
        cross_entropy = cross_entropy / len(keep)
        return cross_entropy