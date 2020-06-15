import torch
import numpy as np


class WeightedCELoss:
    def __init__(self, ignore_index=0, weights_name='own', device='cpu'):
        self._ignore_index = ignore_index
        self._device = device

        if weights_name == 's3dis_freq':
            class_freq = np.asarray([31751233,
                                      28499556,
                                      49263093,
                                      4684592,
                                      4422268,
                                      3693322,
                                      14893175,
                                      6781115,
                                      7746894,
                                      954413,
                                      10256409,
                                      2342048,
                                      24874692])
            class_freq = class_freq / np.sum(class_freq)
            self._class_weights = torch.tensor(-np.log(class_freq), dtype=torch.float, device=self._device)
        elif weights_name == 's3dis_freq_area_1':
            class_freq = np.asarray([44397044,
                                     38754846,
                                     64479250,
                                     2615100,
                                     4230375,
                                     5413617,
                                     10829141,
                                     7646501,
                                     8276232,
                                     991476,
                                     15567431,
                                     2591234,
                                     23727430])

            class_freq = class_freq / np.sum(class_freq)

            self._class_weights = torch.tensor(-np.log(class_freq), dtype=torch.float, device=self._device)
        elif weights_name == 's3dis_freq_area_2':
            class_freq = np.asarray([42709152,
                                     35728700,
                                     63405291,
                                     4342948,
                                     5161507,
                                     6709431,
                                     10532030,
                                     8544499,
                                     5772991,
                                     1009859,
                                     15428959,
                                     2916638,
                                     23969110])

            class_freq = class_freq / np.sum(class_freq)

            self._class_weights = torch.tensor(-np.log(class_freq), dtype=torch.float, device=self._device)
        elif weights_name == 's3dis_freq_area_3':
            class_freq = np.asarray([49023883,
                                     42198367,
                                     70809123,
                                     4388250,
                                     5177941,
                                     6560881,
                                     12198652,
                                     8717549,
                                     8921257,
                                     971584,
                                     16062298,
                                     3106569,
                                     26747960
                                     ])

            class_freq = class_freq / np.sum(class_freq)

            self._class_weights = torch.tensor(-np.log(class_freq), dtype=torch.float, device=self._device)
        elif weights_name == 's3dis_freq_area_4':
            class_freq = np.asarray([45031514,
                                     38202570,
                                     62718036,
                                     4643010,
                                     4608205,
                                     5804196,
                                     10269280,
                                     8064771,
                                     8088850,
                                     895176,
                                     14694951,
                                     3183734,
                                     23872180
                                     ])

            class_freq = class_freq / np.sum(class_freq)

            self._class_weights = torch.tensor(-np.log(class_freq), dtype=torch.float, device=self._device)
        elif weights_name == 's3dis_freq_area_5':
            class_freq = np.asarray([37334028,
                                     32206900,
                                     52921324,
                                     4719832,
                                     4145093,
                                     4127868,
                                     10681455,
                                     6318085,
                                     7930065,
                                     949299,
                                     9209662,
                                     2457821,
                                     21825992
                                     ])

            class_freq = class_freq / np.sum(class_freq)

            self._class_weights = torch.tensor(-np.log(class_freq), dtype=torch.float, device=self._device)
        elif weights_name == 's3dis_freq_area_6':
            class_freq = np.asarray([45068494,
                                     38947597,
                                     65373061,
                                     3002140,
                                     4319279,
                                     5843407,
                                     10819012,
                                     7043625,
                                     7990060,
                                     989111,
                                     15743214,
                                     2701889,
                                     24352543
                                     ])

            class_freq = class_freq / np.sum(class_freq)

            self._class_weights = torch.tensor(-np.log(class_freq), dtype=torch.float, device=self._device)
        elif weights_name == 'matterport_freq':
            # matterport class frequencies
            class_freq = np.asarray([42797343, 68312308, 43961619, 5127719, 3374056, 6806900, 2898291, 3663051,
                                     12716828, 7734540, 976286, 2195565, 850488, 869255, 3739255, 299950, 226355,
                                     359425, 1184173, 740575, 3825545, 49682488])

            class_freq[0] = 0
            class_freq = class_freq / np.sum(class_freq)
            class_freq[0] = 1.0

            self._class_weights = torch.tensor(-np.log(class_freq), dtype=torch.float, device=self._device)
        elif weights_name == 'texturenet':
            # weights used by TextureNet
            class_weights = np.asarray([0.000000000000000000e+00,
                                       3.508061818168880297e+00,
                                       4.415242725535003743e+00,
                                       1.929816058226905895e+01,
                                       2.628740008695115193e+01,
                                       1.212917345982307893e+01,
                                       2.826658055253028934e+01,
                                       2.148932725385034459e+01,
                                       1.769486222014486643e+01,
                                       1.991481374929695747e+01,
                                       2.892054111644061365e+01,
                                       6.634054658350238753e+01,
                                       6.669804496207542854e+01,
                                       3.332619576690268559e+01,
                                       3.076747790368030167e+01,
                                       6.492922584696864874e+01,
                                       7.542849603844955197e+01,
                                       7.551157920875556329e+01,
                                       7.895305324715594963e+01,
                                       7.385072181024294480e+01,
                                       2.166310943989462956e+01])
            self._class_weights = torch.tensor(class_weights, dtype=torch.float, device=self._device)
        elif weights_name == 'None':
            self._class_weights = None
        else:
            raise ValueError(f"Loss weights called {weights_name} do not exist")

        self._loss = torch.nn.CrossEntropyLoss(weight=self._class_weights, ignore_index=self._ignore_index)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value
        if self._class_weights is not None:
            self._class_weights = self._class_weights.to(self._device)
        self._loss = torch.nn.CrossEntropyLoss(weight=self._class_weights, ignore_index=self._ignore_index)

    def __call__(self, output, target):
        return self._loss(output, target)
