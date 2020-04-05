import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import ntpath


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test

    img_dir = os.path.join(web_dir, 'images')
    log_name = os.path.join(img_dir, 'loss_log.txt')
    PSNR = 0
    SSIM = 0


    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        losses = model.get_current_losses()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        message = name
        for k, v in losses.items():
            if k=='PSNR':
                PSNR = PSNR + v
            if k=='SSIM':
                SSIM = SSIM + v
            message += '%s: %.5f ' % (k, v)
        print(message)
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    webpage.save()
    with open(log_name, "a") as log_file:
        log_file.write('averagepsnr%s\n' % (PSNR / i))
        log_file.write('averageSSIM%s\n' % (SSIM / i))
    print('psnr%f'%(PSNR/i))
    print('ssim%f'%(SSIM/i))