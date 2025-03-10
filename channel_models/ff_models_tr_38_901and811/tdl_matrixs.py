"""
File: tdl_matrixs.py

Purpose:
TODO

Authors: Ernesto Fontes Pupo / Claudia Carballo González
         University of Cagliari
Date: 2024-10-30
Version: 1.0.0
                   GNU LESSER GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

    LLSim5G is a link-level simulator for HetNet 5G use cases.
    Copyright (C) 2024  Ernesto Fontes, Claudia Carballo

"""

import numpy as np

# ETSI TR_38.901 v17.0.0 Table 7.7.2-1to5
# 7.7.2 Tapped Delay Line (TDL) models

tdl_cp = {
    #    Delay[ns]
    #    Power[dB]

    "A": np.array(
        [
            [
                0,
                0.3819,
                0.4025,
                0.5868,
                0.4610,
                0.5375,
                0.6708,
                0.5750,
                0.7618,
                1.5375,
                1.8978,
                2.2242,
                2.1717,
                2.4942,
                2.5119,
                3.0582,
                4.0810,
                4.4579,
                4.5695,
                4.7966,
                5.0066,
                5.3043,
                9.6586,
            ],
            [
                -13.4,
                0,
                -2.2,
                -4,
                -6,
                -8.2,
                -9.9,
                -10.5,
                -7.5,
                -15.9,
                -6.6,
                -16.7,
                -12.4,
                -15.2,
                -10.8,
                -11.3,
                -12.7,
                -16.2,
                -18.3,
                -18.9,
                -16.6,
                -19.9,
                -29.7,
            ]
        ]
    ).T,
    "B": np.array(
        [
            [
                0,
                0.1072,
                0.2155,
                0.2095,
                0.2870,
                0.2986,
                0.3752,
                0.5055,
                0.3681,
                0.3697,
                0.5700,
                0.5283,
                1.1021,
                1.2756,
                1.5474,
                1.7842,
                2.0169,
                2.8294,
                3.0219,
                3.6187,
                4.1067,
                4.2790,
                4.7834,
            ],
            [
                0,
                -2.2,
                -4,
                -3.2,
                -9.8,
                -3.2,
                -3.4,
                -5.2,
                -7.6,
                -3,
                -8.9,
                -9,
                -4.8,
                -5.7,
                -7.5,
                -1.9,
                -7.6,
                -12.2,
                -9.8,
                -11.4,
                -14.9,
                -9.2,
                -11.3,
            ]
        ]
    ).T,
    "C": np.array(
        [
            [
                0,
                0.2099,
                0.2219,
                0.2329,
                0.2176,
                0.6366,
                0.6448,
                0.6560,
                0.6584,
                0.7935,
                0.8213,
                0.9336,
                1.2285,
                1.3083,
                2.1704,
                2.7105,
                4.2589,
                4.6003,
                5.4902,
                5.6077,
                6.3065,
                6.6374,
                7.0427,
                8.6523,
            ],
            [
                -4.4,
                -1.2,
                -3.5,
                -5.2,
                -2.5,
                0,
                -2.2,
                -3.9,
                -7.4,
                -7.1,
                -10.7,
                -11.1,
                -5.1,
                -6.8,
                -8.7,
                -13.2,
                -13.9,
                -13.9,
                -15.8,
                -17.1,
                -16,
                -15.7,
                -21.6,
                -22.8,
            ]
        ]
    ).T,
    "D": np.array(
        [
            [
                0,
                0.035,
                0.612,
                1.363,
                1.405,
                1.804,
                2.596,
                1.775,
                4.042,
                7.937,
                9.424,
                9.708,
                12.525,
            ],
            [
                -13.5,
                -18.8,
                -21,
                -22.8,
                -17.9,
                -20.1,
                -21.9,
                -22.9,
                -27.8,
                -23.6,
                -24.8,
                -30.0,
                -27.7,
            ]
        ]
    ).T,
    "E": np.array(
        [
            [
                0,
                0.5133,
                0.5440,
                0.5630,
                0.5440,
                0.7112,
                1.9092,
                1.9293,
                1.9589,
                2.6426,
                3.7136,
                5.4524,
                12.0034,
                20.6519,
            ],
            [
                -22.03,
                -15.8,
                -18.1,
                -19.8,
                -22.9,
                -22.4,
                -18.6,
                -20.8,
                -22.6,
                -22.3,
                -25.6,
                -20.2,
                -29.8,
                -29.2,
            ]
        ]
    ).T,

# ETSI TR_38.811, Table 6.9.2-1
# Tapped Delay Line (TDL) models

    "A_ntn": np.array(
        [
            [
                0,
                1.0811,
                2.8416,
            ],
            [
                0.000,
                -4.675,
                -6.482,
            ]
        ]
    ).T,
    "B_ntn": np.array(
        [
            [
                0,
                0.7249,
                0.7410,
                5.7392,
            ],
            [
                0,
                -1.973,
                -4.332,
                -11.914,
            ]
        ]
    ).T,
    "C_ntn": np.array(
        [
            [
                0,
                0,
                14.8124,
            ],
            [
                -0.394,
                -10.618,
                -23.373,
            ]
        ]
    ).T,
    "D_ntn": np.array(
        [
            [
                0,
                0,
                0.5596,
                7.3340,
            ],
            [
                -0.284,
                -11.991,
                -9.887,
                -16.771,
            ]
        ]
    ).T,

# ETSI TR_38.181 Table G.2.1.1-2

    "A100_ntn": np.array(
        [
            [
                0,
                110,
                285,
            ],
            [
                0,
                -4.7,
                -6.5,
            ]
        ]
    ).T,
    "C5_ntn": np.array(
        [
            [
                0,
                0,
                60,
            ],
            [
                -0.6,
                -8.9,
                -21.5,
            ]
        ]
    ).T
}

tdl_rice_factors = {
    "A": np.zeros(tdl_cp["A"][:,0].shape),
    "B": np.zeros(tdl_cp["B"][:,0].shape),
    "C": np.zeros(tdl_cp["C"][:,0].shape),
    "D": np.zeros(tdl_cp["D"][:,0].shape),
    "E": np.zeros(tdl_cp["E"][:,0].shape),
    "A_ntn": np.zeros(tdl_cp["A"][:, 0].shape),
    "B_ntn": np.zeros(tdl_cp["B"][:, 0].shape),
    "C_ntn": np.zeros(tdl_cp["C"][:, 0].shape),
    "D_ntn": np.zeros(tdl_cp["D"][:, 0].shape),
    "A100_ntn": np.zeros(tdl_cp["E"][:, 0].shape),
    "C5_ntn": np.zeros(tdl_cp["E"][:, 0].shape)
}
tdl_rice_factors["D"][0] = 13.3
tdl_rice_factors["E"][0] = 22.3
tdl_rice_factors["C_ntn"][0] = 10.224
tdl_rice_factors["D_ntn"][0] = 11.707

tdl_los_doppler_frequency = {
    "D": 0.7,
    "E": 0.7
}

# print(tdl_rice_factors["D"])
# print(tdl_cp["D"][:,0].shape)


# ETSI TR_38.811, Table 6.9.2-1
# Tapped Delay Line (TDL) models

ntn_tdl_cp = {
    #    Delay[ns]
    #    Power[dB]

    "A": np.array(
        [
            [
                0,
                1.0811,
                2.8416,
            ],
            [
                0.000,
                -4.675,
                -6.482,
            ]
        ]
    ).T,
    "B": np.array(
        [
            [
                0,
                0.7249,
                0.7410,
                5.7392,
            ],
            [
                0,
                -1.973,
                -4.332,
                -11.914,
            ]
        ]
    ).T,
    "C": np.array(
        [
            [
                0,
                0,
                14.8124,
            ],
            [
                -0.394,
                -10.618,
                -23.373,
            ]
        ]
    ).T,
    "D": np.array(
        [
            [
                0,
                0,
                0.5596,
                7.3340,
            ],
            [
                -0.284,
                -11.991,
                -9.887,
                -16.771,
            ]
        ]
    ).T,

# ETSI TR_38.181 Table G.2.1.1-2

    "A100": np.array(
        [
            [
                0,
                110,
                285,
            ],
            [
                0,
                -4.7,
                -6.5,
            ]
        ]
    ).T,
    "C5": np.array(
        [
            [
                0,
                0,
                60,
            ],
            [
                -0.6,
                -8.9,
                -21.5,
            ]
        ]
    ).T
}