import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
import torch
from cost_matric import compute_cost_matrix, hierarchy, leaf_labels
import ot
import ot.plot


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.weight = nn.Linear(self.config.num_labels,1)
        self.init_weights()
        self.M = compute_cost_matrix(hierarchy, leaf_labels)
        self.M_tensor = torch.tensor(self.M, dtype=torch.float32) 
        self.softmax = nn.Softmax(dim=1)
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits_soft = self.softmax(logits)
        x = self.weight(logits)
        x = torch.sigmoid(x)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        assert len(outputs) < 3

        if labels is not None:
            total_loss = torch.tensor(0.0).to(device) 
            batch_size = logits_soft.size(0) 

            for i in range(batch_size):
                single_logits_soft = logits_soft[i]
                single_label = labels[i]

                if single_label.sum() > 0: 
                    single_label = single_label / single_label.sum()

                loss = ot.emd2(single_logits_soft, single_label, self.M_tensor)
                total_loss += loss  

            average_loss = total_loss / batch_size 
            BCE_loss = self.BCE(logits, labels)

            final_loss = BCE_loss *(1-x) + average_loss * x

            list = [final_loss, outputs, average_loss, BCE_loss]
            outputs = list

        return outputs  # (loss), logits, (hidden_states), (attentions)
