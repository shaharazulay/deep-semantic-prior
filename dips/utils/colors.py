from torch.autograd import Variable

def rgb_to_yiq(input_):
    """

    :param input_: of the form [1, 3, ., .]
    :return:
    """
    output = Variable(input_.data.new(*input_.size()))
    output[:, 0, :, :] = 0.299 * input_[:, 0, :, :] + 0.587 * input_[:, 1, :, :] + 0.114 * input_[:, 2, :, :]
    output[:, 1, :, :] = 0.59590059 * input_[:, 0, :, :] - 0.27455667 * input_[:, 1, :, :] - 0.32134392 * input_[:, 2, :, :]
    output[:, 2, :, :] = 0.21153661 * input_[:, 0, :, :] - 0.52273617 * input_[:, 1, :, :] + 0.31119955 * input_[:, 2, :, :]
    return output