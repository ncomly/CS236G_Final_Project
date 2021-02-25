# Inspired by and partially taken from CS 236G Coursera Course Content
from utils import *
from model import *
from data import *


# runtime params
adv_criterion = nn.MSELoss() 
recon_criterion = nn.L1Loss() 

dim_A = 3
dim_B = 3
load_shape = 286
target_shape = 256
    

## Main
def main(args):
    ## Load Dataset
    # create transform
    transform = transforms.Compose([
        transforms.Resize(load_shape),
        transforms.RandomCrop(target_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # get dataset
    dataset = ImageDataset(args.data_folder, 
                            transform=transform, 
                            a_subroot=args.A_subfolder, 
                            b_subroot=args.B_subfolder)

    ## Create Generator and Discriminator
    gen_AB = Generator(dim_A, dim_B).to(args.device)
    gen_BA = Generator(dim_B, dim_A).to(args.device)
    gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=args.lr, betas=(0.5, 0.999))
    disc_A = Discriminator(dim_A).to(args.device)
    disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    disc_B = Discriminator(dim_B).to(args.device)
    disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    ## Initialize weights
    if args.checkpoint:
        print(f'Loading pretrained model: {args.checkpoint}')
        args.save_path = args.checkpoint.replace('.pth', '') + '_'
        print(f'Save path overwritten to {args.save_path}XXX.pth')
        pre_dict = torch.load(args.checkpoint)
        gen_AB.load_state_dict(pre_dict['gen_AB'])
        gen_BA.load_state_dict(pre_dict['gen_BA'])
        gen_opt.load_state_dict(pre_dict['gen_opt'])
        disc_A.load_state_dict(pre_dict['disc_A'])
        disc_A_opt.load_state_dict(pre_dict['disc_A_opt'])
        disc_B.load_state_dict(pre_dict['disc_B'])
        disc_B_opt.load_state_dict(pre_dict['disc_B_opt'])
    else:
        if args.save:
            args.save_path += 'cycleGAN_'
            print(f'Model will be saved to {args.save_path}XXX.pth')
        gen_AB = gen_AB.apply(weights_init)
        gen_BA = gen_BA.apply(weights_init)
        disc_A = disc_A.apply(weights_init)
        disc_B = disc_B.apply(weights_init)

    # Train

    if args.train:
        # Tensorboard summary writer
        writer = SummaryWriter()

        mean_generator_loss = 0
        mean_discriminator_loss = 0
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        cur_step = 0

        for epoch in range(args.epochs):
            print(f'Epoch {epoch}/{args.epochs}')
            # Dataloader returns the batches
            for real_A, real_B in tqdm(dataloader):
                real_A = nn.functional.interpolate(real_A, size=target_shape)
                real_B = nn.functional.interpolate(real_B, size=target_shape)
                cur_batch_size = len(real_A)
                real_A = real_A.to(args.device)
                real_B = real_B.to(args.device)

                ### Update discriminator A ###
                disc_A_opt.zero_grad() # Zero out the gradient before backpropagation
                with torch.no_grad():
                    fake_A = gen_BA(real_B)
                disc_A_loss = get_disc_loss(real_A, fake_A, disc_A, adv_criterion)
                disc_A_loss.backward(retain_graph=True) # Update gradients
                disc_A_opt.step() # Update optimizer

                ### Update discriminator B ###
                disc_B_opt.zero_grad() # Zero out the gradient before backpropagation
                with torch.no_grad():
                    fake_B = gen_AB(real_A)
                disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion)
                disc_B_loss.backward(retain_graph=True) # Update gradients
                disc_B_opt.step() # Update optimizer

                ### Update generator ###
                gen_opt.zero_grad()
                gen_loss, fake_A, fake_B = get_gen_loss(
                    real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, recon_criterion, recon_criterion
                )
                gen_loss.backward() # Update gradients
                gen_opt.step() # Update optimizer

                # Keep track of the average discriminator loss
                mean_discriminator_loss += disc_A_loss.item() / args.write_step
                # Keep track of the average generator loss
                mean_generator_loss += gen_loss.item() / args.write_step

                ### Tensorboard ###
                if cur_step % args.write_step == 0:
                    writer.add_scalar("Mean Generator Loss", mean_generator_loss, cur_step)
                    writer.add_scalar("Mean Discriminator Loss", mean_discriminator_loss, cur_step)

                    mean_generator_loss = 0
                    mean_discriminator_loss = 0

                ## Save Images ##
                if cur_step % args.display_step == 0:
                    writer.add_image('Real AB', convert_tensor_images(torch.cat([real_A, real_B], dim=-1), size=(dim_A, target_shape, target_shape)))
                    writer.add_image('Fake BA', convert_tensor_images(torch.cat([fake_B, fake_A], dim=-1), size=(dim_A, target_shape, target_shape)))
                    writer.flush()

                ## Model Saving ##
                if args.save and cur_step % args.save_step == 0:
                    torch.save({
                        'gen_AB': gen_AB.state_dict(),
                        'gen_BA': gen_BA.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc_A': disc_A.state_dict(),
                        'disc_A_opt': disc_A_opt.state_dict(),
                        'disc_B': disc_B.state_dict(),
                        'disc_B_opt': disc_B_opt.state_dict()
                    }, f"{args.save_path}{cur_step}.pth")
                cur_step += 1


        writer.flush()


if __name__ == '__main__':

    # get arguments
    args = parse_args()
    
    main(args)
