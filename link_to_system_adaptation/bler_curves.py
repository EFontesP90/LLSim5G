import numpy as np


class Bler_vs_Sinr_curves(object):
    """
    07/05/2024
    Bler vs SINR curves for the AWGN chanel and the real channel implemented in simu5G omnet++.

    Required attributes: (channel_type)

    """

    def __init__(self, channel_type):

        self.channel_type = channel_type  # A string value that could be: awgn, real

    def awgn_curves(self):

        bler = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9998, 0.9998, 0.9998, 0.9988,
                                      0.9984, 0.9956, 0.9862, 0.9732, 0.9384, 0.8938, 0.8144, 0.7088, 0.5742, 0.4392,
                                      0.2864, 0.1818, 0.0988, 0.0476, 0.0192, 0.0086, 0.0022, 0.0004, 0, 0, 0, 0, 0, 0,
                                      ], [
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9998, 1, 0.9998, 0.999,
                                         0.9964, 0.9906, 0.9724, 0.9188, 0.8558, 0.7332, 0.5684, 0.382, 0.2232, 0.1092,
                                         0.0486, 0.0124, 0.0048, 0.0018, 0.0002, 0, 0, 0, 0, 0, 0, 0,
                                     ], [
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9996, 0.9996,
                                         0.9968, 0.9834, 0.946, 0.8518, 0.699, 0.477, 0.2664, 0.1112, 0.041, 0.0092,
                                         0.0012, 0, 0.0002, 0, 0, 0, 0, 0, 0, 0,
                                     ], [
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9998,
                                         0.998, 0.987, 0.9296, 0.779, 0.5356, 0.2718, 0.0902, 0.0192, 0.0032, 0.001, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0,
                                     ], [
                                         1, 1, 1, 1, 1, 1, 0.9998, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0.9956, 0.9736, 0.8534, 0.5952, 0.2762, 0.0702, 0.011, 0.0018, 0.0002, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0,
                                     ], [
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9996,
                                         0.9958, 0.9694, 0.8184, 0.4972, 0.176, 0.03, 0.0026, 0.0002, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0,
                                     ], [
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         1, 1, 1, 1, 1, 1, 1, 0.9988, 0.9842, 0.8918, 0.5888, 0.2236, 0.0402, 0.0028, 0,
                                         0, 0,
                                     ], [
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9998,
                                         0.9996, 0.9958, 0.9434, 0.6908, 0.2936, 0.0522, 0.0032, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0,
                                     ], [
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0.9926, 0.9452, 0.7416, 0.3608, 0.0642, 0.0044, 0.0008, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0,
                                     ], [
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9996,
                                         0.9842, 0.8498, 0.4582, 0.1058, 0.0084, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     ], [
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9994,
                                         0.974, 0.7564, 0.3054, 0.0488, 0.0034, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     ], [
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0.9906, 0.9052, 0.567, 0.1548, 0.015, 0.0002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     ], [
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         1, 0.995, 0.8998, 0.5164, 0.1344, 0.018, 0.0022, 0.0002, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0,
                                     ], [
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         1, 0.9996, 0.951, 0.6506, 0.1864, 0.016, 0.0012, 0.0008, 0.0002, 0.0002,
                                         0.0002, 0, 0.0002, 0, 0, 0, 0,
                                     ], [
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         1, 1, 0.9968, 0.955, 0.6856, 0.2942, 0.072, 0.0166, 0.0034, 0.001, 0.0002, 0,
                                         0.0002, 0, 0, 0, 0,
                                     ]])

        sinr = np.array([[-14.5, -14.25, -14, -13.75, -13.5, -13.25, -13, -12.75, -12.5, -12.25, -12,
                                      -11.75, -11.5, -11.25, -11, -10.75, -10.5, -10.25, -10, -9.75, -9.5, -9.25, -9,
                                      -8.75, -8.5, -8.25, -8, -7.75, -7.5, -7.25, -7, -6.75, -6.5, -6.25, -6, -5.75,
                                      -5.5, -5.25, -5, -4.75, -4.5, -4.25, -4,
                                      ], [
                                         -12.5, -12.25, -12, -11.75, -11.5, -11.25, -11, -10.75, -10.5, -10.25, -10,
                                         -9.75, -9.5, -9.25, -9, -8.75, -8.5, -8.25, -8, -7.75, -7.5, -7.25, -7, -6.75,
                                         -6.5, -6.25, -6, -5.75, -5.5, -5.25, -5, -4.75, -4.5, -4.25, -4, -3.75, -3.5,
                                         -3.25, -3, -2.75, -2.5, -2.25, -2,
                                     ], [
                                         -10.5, -10.25, -10, -9.75, -9.5, -9.25, -9, -8.75, -8.5, -8.25, -8, -7.75,
                                         -7.5, -7.25, -7, -6.75, -6.5, -6.25, -6, -5.75, -5.5, -5.25, -5, -4.75, -4.5,
                                         -4.25, -4, -3.75, -3.5, -3.25, -3, -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25,
                                         -1, -0.75, -0.5, -0.25, 0,
                                     ], [
                                         -8.5, -8.25, -8, -7.75, -7.5, -7.25, -7, -6.75, -6.5, -6.25, -6, -5.75, -5.5,
                                         -5.25, -5, -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3, -2.75, -2.5, -2.25,
                                         -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25,
                                         1.5, 1.75, 2,
                                     ], [
                                         -6.5, -6.25, -6, -5.75, -5.5, -5.25, -5, -4.75, -4.5, -4.25, -4, -3.75, -3.5,
                                         -3.25, -3, -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25,
                                         0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75,
                                         4,
                                     ], [
                                         -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3, -2.75, -2.5, -2.25, -2, -1.75, -1.5,
                                         -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25,
                                         2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6,
                                     ], [
                                         -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3, -2.75, -2.5, -2.25, -2, -1.75, -1.5,
                                         -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25,
                                         2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6,
                                     ], [
                                         -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3,
                                         3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7,
                                         7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10,
                                     ], [
                                         1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25,
                                         5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25,
                                         9.5, 9.75, 10, 10.25, 10.5, 10.75, 11, 11.25, 11.5, 11.75, 12,
                                     ], [
                                         3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25,
                                         7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10, 10.25, 10.5, 10.75, 11,
                                         11.25, 11.5, 11.75, 12, 12.25, 12.5, 12.75, 13, 13.25, 13.5, 13.75, 14,
                                     ], [
                                         5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25,
                                         9.5, 9.75, 10, 10.25, 10.5, 10.75, 11, 11.25, 11.5, 11.75, 12, 12.25, 12.5,
                                         12.75, 13, 13.25, 13.5, 13.75, 14, 14.25, 14.5, 14.75, 15, 15.25, 15.5, 15.75,
                                         16,
                                     ], [
                                         7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10, 10.25, 10.5,
                                         10.75, 11, 11.25, 11.5, 11.75, 12, 12.25, 12.5, 12.75, 13, 13.25, 13.5, 13.75,
                                         14, 14.25, 14.5, 14.75, 15, 15.25, 15.5, 15.75, 16, 16.25, 16.5, 16.75, 17,
                                         17.25, 17.5,
                                     ], [
                                         8.5, 8.75, 9, 9.25, 9.5, 9.75, 10, 10.25, 10.5, 10.75, 11, 11.25, 11.5, 11.75,
                                         12, 12.25, 12.5, 12.75, 13, 13.25, 13.5, 13.75, 14, 14.25, 14.5, 14.75, 15,
                                         15.25, 15.5, 15.75, 16, 16.25, 16.5, 16.75, 17, 17.25, 17.5, 17.75, 18, 18.25,
                                         18.5, 18.75, 19,
                                     ], [
                                         10.25, 10.5, 10.75, 11, 11.25, 11.5, 11.75, 12, 12.25, 12.5, 12.75, 13, 13.25,
                                         13.5, 13.75, 14, 14.25, 14.5, 14.75, 15, 15.25, 15.5, 15.75, 16, 16.25, 16.5,
                                         16.75, 17, 17.25, 17.5, 17.75, 18, 18.25, 18.5, 18.75, 19, 19.25, 19.5, 19.75,
                                         20, 20.25, 20.5, 20.75,
                                     ], [
                                         12, 12.25, 12.5, 12.75, 13, 13.25, 13.5, 13.75, 14, 14.25, 14.5, 14.75, 15,
                                         15.25, 15.5, 15.75, 16, 16.25, 16.5, 16.75, 17, 17.25, 17.5, 17.75, 18, 18.25,
                                         18.5, 18.75, 19, 19.25, 19.5, 19.75, 20, 20.25, 20.5, 20.75, 21, 21.25, 21.5,
                                         21.75, 22, 22.25, 22.5,
                                     ]])

        return bler, sinr

    def real_curves_simu5g(self):

        bler = np.array([[
            1, 1, 0.996, 0.992, 0.968, 0.88, 0.76, 0.564, 0.364, 0.22, 0.084, 0.044, 0.008, 0, 0.00, 0,
            # penultimo era 0.004 yo lo cambie a cero
        ], [
            1, 1, 1, 0.996, 0.968, 0.928, 0.84, 0.68, 0.42, 0.284, 0.12, 0.076, 0.02, 0.008, 0, 0,
        ], [
            1, 1, 1, 0.992, 0.956, 0.924, 0.752, 0.588, 0.436, 0.3, 0.156, 0.048, 0.032, 0, 0, 0,
        ], [
            1, 1, 1, 0.988, 0.984, 0.9, 0.784, 0.596, 0.432, 0.248, 0.112, 0.072, 0.008, 0.002, 0, 0,
            # antepenultimo era 0.02 yo lo cambie a 0.002
        ], [
            1, 1, 0.988, 0.996, 0.924, 0.844, 0.724, 0.46, 0.356, 0.208, 0.076, 0.02, 0.008, 0, 0, 0,
        ], [
            1, 1, 0.996, 0.984, 0.936, 0.868, 0.792, 0.66, 0.424, 0.304, 0.164, 0.064, 0.036, 0.016, 0.004, 0,
        ], [
            1, 1, 1, 0.996, 0.98, 0.928, 0.836, 0.672, 0.544, 0.304, 0.156, 0.08, 0.036, 0.008, 0.004, 0,
        ], [
            1, 1, 1, 0.988, 0.98, 0.936, 0.836, 0.676, 0.476, 0.344, 0.18, 0.088, 0.028, 0.012, 0, 0,
        ], [
            1, 1, 1, 0.992, 0.968, 0.944, 0.832, 0.668, 0.58, 0.404, 0.288, 0.164, 0.084, 0.016, 0.012, 0,
        ], [
            1, 1, 1, 0.992, 0.992, 0.96, 0.876, 0.728, 0.552, 0.376, 0.216, 0.132, 0.036, 0.024, 0.008, 0.0,
        ], [
            1, 1, 1, 1, 0.992, 0.976, 0.916, 0.876, 0.716, 0.664, 0.396, 0.3, 0.16, 0.076, 0.06, 0.02,
        ], [
            1, 1, 1, 1, 1, 0.996, 0.972, 0.908, 0.812, 0.696, 0.504, 0.312, 0.24, 0.18, 0.088, 0.008,
        ], [
            1, 1, 1, 0.98, 0.952, 0.88, 0.804, 0.68, 0.516, 0.368, 0.188, 0.14, 0.068, 0.052, 0.028, 0.02,
        ], [
            0.98, 0.968, 0.944, 0.912, 0.8, 0.736, 0.572, 0.42, 0.332, 0.26, 0.16, 0.144, 0.068, 0.036, 0.02, 0.004,
        ], [
            0.964, 0.932, 0.848, 0.78, 0.668, 0.632, 0.488, 0.452, 0.372, 0.296, 0.164, 0.124, 0.072, 0.048, 0.032,
            0.02,
        ]])

        sinr = np.array([
            [
                -14.5, -13.5, -12.5, -11.5, -10.5, -9.5, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5,
            ], [
                -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2,
            ], [
                -10.5, -9.5, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5,
            ], [
                -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
            ], [
                -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            ], [
                -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            ], [
                -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            ], [
                0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5,
            ], [
                2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5,
            ], [
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            ], [
                6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            ], [
                7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5,
            ], [
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            ], [
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            ], [
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
            ]])

        return bler, sinr

    def get_channel_curves(self):
        if self.channel_type == "awgn":
            bler, sinr = self.awgn_curves()
        elif self.channel_type == "real":
            bler, sinr = self.real_curves_simu5g()
        return bler, sinr


def get_channel_curves(channel_type):
    curves = Bler_vs_Sinr_curves(channel_type)
    bler, sinr = curves.get_channel_curves()
    return bler, sinr

