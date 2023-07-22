import models.modules.Ours as Ours
import models.modules.flownet as flownet



####################
# define network
####################
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'Ours':
        netG = Ours.FGDCN(nf=opt_net['nf'], opt=opt)
    
    elif which_model == 'flownet':
        netG = flownet.IFNet()


    
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
