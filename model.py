import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
import torch
from cost_matric import compute_cost_matrix, hierarchy, leaf_labels
import ot
import ot.plot


class BertForMultiLabelClassification(BertPreTrainedModel):
    """
    BERT-based model for multi-label emotion classification with hierarchical loss.
    
    This model extends BERT for multi-label classification and incorporates:
    - Softmax over logits (for multi-label classification)
    - Earth Mover's Distance (EMD) loss based on hierarchical emotion structure
    - Combined BCE and EMD loss with learnable weighting
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # BERT encoder
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        # Learnable weight for combining BCE and EMD losses
        self.weight = nn.Linear(self.config.num_labels, 1)
        self.init_weights()
        
        # Compute cost matrix based on hierarchical emotion structure
        # The cost matrix encodes distances between emotions in the hierarchy
        self.M = compute_cost_matrix(hierarchy, leaf_labels)
        self.M_tensor = torch.tensor(self.M, dtype=torch.float32)
        
        # Softmax for probability distribution (used with EMD loss)
        self.softmax = nn.Softmax(dim=1)
        # Binary Cross Entropy loss for multi-label classification
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
        
        # Apply softmax to get probability distribution for EMD calculation
        logits_soft = self.softmax(logits)
        
        # Compute learnable weight for loss combination
        x = self.weight(logits)
        x = torch.sigmoid(x)  # Ensure weight is between 0 and 1
        
        # Get device from model parameters
        device = next(self.parameters()).device

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        assert len(outputs) < 3

        if labels is not None:
            total_loss = torch.tensor(0.0).to(device) 
            batch_size = logits_soft.size(0) 
            
            # Ensure M_tensor is on the correct device
            M_tensor = self.M_tensor.to(device)

            # Compute EMD loss for each sample in the batch
            # EMD measures the distance between predicted and true emotion distributions
            # using the hierarchical cost matrix
            for i in range(batch_size):
                single_logits_soft = logits_soft[i]
                single_label = labels[i]

                # Normalize label to probability distribution
                if single_label.sum() > 0: 
                    single_label = single_label / single_label.sum()

                # Compute Earth Mover's Distance
                loss = ot.emd2(single_logits_soft, single_label, M_tensor)
                total_loss += loss  

            # Average EMD loss over batch
            average_loss = total_loss / batch_size 
            # Binary Cross Entropy loss
            BCE_loss = self.BCE(logits, labels)

            # Combine BCE and EMD losses with learnable weighting
            # This is the core innovation: adaptive combination of classification and hierarchical losses
            final_loss = BCE_loss * (1 - x) + average_loss * x

            list = [final_loss, outputs, average_loss, BCE_loss]
            outputs = list

        return outputs  # (loss), logits, (hidden_states), (attentions)
