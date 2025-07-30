import torch
import torch.nn as nn
import torch.nn.functional as F

clip_vis = 'RN50'
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)


class AlignLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2', maps=32, learning_prompt=False):
        super(TeCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        self.clip_model, _ = clip.load(clip_vis)
        self.fc = nn.Linear(1024, maps * 2)

    def forward(self, features, labels, grade=5):

        n = features.shape[0]
        expanded_features1 = features.unsqueeze(1).expand(n, n, 32)
        expanded_features2 = features.unsqueeze(0).expand(n, n, 32)

        feature_diff = torch.cat([expanded_features1, expanded_features2], dim=2)

        diag_mask = ~torch.eye(n, dtype=torch.bool, device=feature_diff.device)

        features_copy = feature_diff.clone()

        feature_diff = features_copy.masked_select(diag_mask.unsqueeze(2)).view(n, n - 1, -1)
        text_fs = []

        text0 = "The directions of gaze in the two faces in the photo are identical."
        text1 = "The directions of gaze in the two faces in the photo are highly similar."
        text2 = "The directions of gaze in the two faces in the photo are moderately similar."
        text3 = "The directions of gaze in the two faces in the photo are slightly similar."
        text4 = "The directions of gaze in the two faces in the photo are not similar."
        text5 = "The directions of gaze in the two faces in the photo are similar."
        if grade == 5:
            texts = [text0, text1, text2, text3, text4]
        elif grade == 3:
            texts = [text0, text5, text4]
        elif grade == 2:
            texts = [text5, text4]

        with torch.no_grad():
            self.clip_model.eval()
            for i in range(0, grade):
                text = clip.tokenize(texts[i]).cuda()
                text_f = self.clip_model.encode_text(text) 
                text_f /= text_f.norm(dim=-1, keepdim=True)
                text_f = self.fc(text_f.to(torch.float32))
                text_f = text_f.repeat(n, n-1, 1)
                text_fs.append(text_f)

        label_diffs = torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)  

        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(label_diffs.device)).bool()).view(n, n - 1)
        masks = []
        if grade == 5:
            mask0 = torch.where(label_diffs < 0.1, 1.0, 0.0)
            mask1 = torch.where((label_diffs < 0.2) & (label_diffs >= 0.1), 1.0, 0.0)
            mask2 = torch.where((0.3 > label_diffs) & (label_diffs >= 0.2), 1.0, 0.0)
            mask3 = torch.where((0.5 > label_diffs) & (label_diffs >= 0.3), 1.0, 0.0)
            mask4 = torch.where(label_diffs >= 0.5, 1.0, 0.0)
            masks = [mask0, mask1, mask2, mask3, mask4]
        elif grade == 3:
            mask0 = torch.where(label_diffs < 0.1, 1.0, 0.0)
            mask1 = torch.where((label_diffs < 0.2) & (label_diffs >= 0.1), 1.0, 0.0)
            mask2 = torch.where(label_diffs >= 0.2, 1.0, 0.0)
            masks = [mask0, mask1, mask2]
        elif grade == 2:
            mask0 = torch.where(label_diffs < 0.2, 1.0, 0.0)
            mask2 = torch.where(label_diffs >= 0.2, 1.0, 0.0)
            masks = [mask0, mask2]

        loss = 0.
        for j in range(0, grade):
            logits = torch.cosine_similarity(feature_diff, text_fs[j], dim=-1)
            loss = loss + nn.functional.cross_entropy(logits, masks[j].cuda())

        loss = loss / grade
        loss = loss * 32 / n

        return loss

