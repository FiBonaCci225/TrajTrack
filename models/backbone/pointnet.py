from mmengine.registry import MODELS
import torch.nn as nn
import torch


@MODELS.register_module()
class PointNet(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.SyncBatchNorm(64, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.SyncBatchNorm(64, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.SyncBatchNorm(128, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.SyncBatchNorm(1024, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.AdaptiveMaxPool1d(1),
        )

    def forward(self, inputs):
        return super().forward(inputs.transpose(1, 2).contiguous())

@MODELS.register_module()
class MiniPointNet(nn.Module):

    def __init__(self, input_channel, per_point_mlp, hidden_mlp, output_size=0):
        """

        :param input_channel: int
        :param per_point_mlp: list
        :param hidden_mlp: list
        :param output_size: int, if output_size <=0, then the final fc will not be used
        """
        super(MiniPointNet, self).__init__()
        seq_per_point = []
        in_channel = input_channel
        for out_channel in per_point_mlp:
            seq_per_point.append(nn.Conv1d(in_channel, out_channel, 1))
            seq_per_point.append(nn.BatchNorm1d(out_channel))
            seq_per_point.append(nn.ReLU())
            in_channel = out_channel
        seq_hidden = []
        for out_channel in hidden_mlp:
            seq_hidden.append(nn.Linear(in_channel, out_channel))
            seq_hidden.append(nn.BatchNorm1d(out_channel))
            seq_hidden.append(nn.ReLU())
            in_channel = out_channel

        # self.per_point_mlp = nn.Sequential(*seq)
        # self.pooling = nn.AdaptiveMaxPool1d(output_size=1)
        # self.hidden_mlp = nn.Sequential(*seq_hidden)

        self.features = nn.Sequential(*seq_per_point,
                                      nn.AdaptiveMaxPool1d(output_size=1),
                                      nn.Flatten(),
                                      *seq_hidden)
        self.output_size = output_size
        if output_size >= 0:
            self.fc = nn.Linear(in_channel, output_size)

    def forward(self, x):
        """

        :param x: B,C,N
        :return: B,output_size
        """

        # x = self.per_point_mlp(x)
        # x = self.pooling(x)
        # x = self.hidden_mlp(x)
        x = self.features(x)
        if self.output_size > 0:
            x = self.fc(x)
        return x

@MODELS.register_module()
class SegPointNet(nn.Module):

    def __init__(self, input_channel, per_point_mlp1, per_point_mlp2, output_size=0, return_intermediate=False):
        """

        :param input_channel: int
        :param per_point_mlp: list
        :param hidden_mlp: list
        :param output_size: int, if output_size <=0, then the final fc will not be used
        """
        super(SegPointNet, self).__init__()
        self.return_intermediate = return_intermediate
        self.seq_per_point = nn.ModuleList()
        in_channel = input_channel
        for out_channel in per_point_mlp1:
            self.seq_per_point.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, out_channel, 1),
                    # nn.Conv1d的输入应当是一个3D的张量，[batch_size, in_channels, input_length]
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU()
                ))
            in_channel = out_channel

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.seq_per_point2 = nn.ModuleList()
        in_channel = in_channel + per_point_mlp1[1]
        for out_channel in per_point_mlp2:
            self.seq_per_point2.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, out_channel, 1),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU()
                ))
            in_channel = out_channel

        self.output_size = output_size
        if output_size >= 0:
            self.fc = nn.Conv1d(in_channel, output_size, 1)  # 把per_point_mlp2的最后一维直接打成2

    def forward(self, x):
        """

        :param x: B,C,N
        :return: B,output_size,N
        """
        second_layer_out = None
        for i, mlp in enumerate(self.seq_per_point):
            x = mlp(x)
            if i == 1:
                second_layer_out = x
        pooled_feature = self.pool(x)  # B,C,1
        pooled_feature_expand = pooled_feature.expand_as(x)
        x = torch.cat([second_layer_out, pooled_feature_expand], dim=1)
        for mlp in self.seq_per_point2:
            x = mlp(x)
        if self.output_size > 0:
            x = self.fc(x)
        if self.return_intermediate:
            return x, pooled_feature.squeeze(dim=-1)
        return x.float()

@MODELS.register_module()
class FeaturePointNet(nn.Module):

    def __init__(self, input_channel, per_point_mlp1, per_point_mlp2, output_size=0, return_intermediate=False):
        super(FeaturePointNet, self).__init__()
        self.return_intermediate = return_intermediate
        self.seq_per_point = nn.ModuleList()
        in_channel = input_channel
        for out_channel in per_point_mlp1:
            self.seq_per_point.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, out_channel, 1),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU()
                ))
            in_channel = out_channel

        self.pre_pool = nn.AdaptiveMaxPool1d(output_size=output_size)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.seq_per_point2 = nn.ModuleList()
        in_channel = in_channel + per_point_mlp1[1]
        for out_channel in per_point_mlp2:
            self.seq_per_point2.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, out_channel, 1),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU()
                ))
            in_channel = out_channel

        self.output_size = output_size
        if output_size >= 0:
            self.fc = nn.Conv1d(in_channel, output_size, 1)

    def forward(self, x):
        """

        :param x: B,C,N
        :return: B,output_size,N
        """
        second_layer_out = None
        for i, mlp in enumerate(self.seq_per_point):
            x = mlp(x)
            if i == 1:
                second_layer_out = x
        pooled_feature = self.pool(x)
        pre_pooled_feature = self.pre_pool(second_layer_out)  # Pool to a fixed size

        pooled_feature_expand = pooled_feature.expand(pooled_feature.shape[0], pooled_feature.shape[1],
                                                      pre_pooled_feature.shape[2])

        x = torch.cat([pre_pooled_feature, pooled_feature_expand], dim=1)
        for mlp in self.seq_per_point2:
            x = mlp(x)
        if self.output_size > 0:
            x = self.fc(x)
        if self.return_intermediate:
            return x, pooled_feature.squeeze(dim=-1)
        return x