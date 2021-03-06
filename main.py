# Inspired by and partially taken from CS 236G Coursera Course Content
from utils import *
from model import *
from data import *


# runtime params
dim_A = 3
dim_B = 3
dim_L = 6
load_shape = 100
target_shape = 100
    

## Main
def main(args):
    # helper function for getting validation examples
    def get_val_examples():
        while True:
            for example in dataloader_val:
                yield example
    # helper for saving the model
    def save_model():
        torch.save({
            'gen_AB': gen_AB.state_dict(),
            'gen_BA': gen_BA.state_dict(),
            'gen_opt': gen_opt.state_dict(),
            'disc_A': disc_A.state_dict(),
            'disc_A_opt': disc_A_opt.state_dict(),
            'disc_B': disc_B.state_dict(),
            'disc_B_opt': disc_B_opt.state_dict()
        }, f"{args.save_path}{epoch}.pth")





    ## Load Dataset
    # create transform
    transform = transforms.Compose([
        transforms.Resize(load_shape),
        transforms.RandomCrop(target_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    if args.train:
        # get dataset
        dataset_train = ImageDataset(args.data_folder, 
                                    transform=transform, 
                                    a_subroot=args.A_subfolder, 
                                    b_subroot=args.B_subfolder,
                                    l_subroot=args.L_subfolder,
                                    mode='train')
        dataset_val   = ImageDataset(args.data_folder, 
                                    transform=transform, 
                                    a_subroot=args.A_subfolder, 
                                    b_subroot=args.B_subfolder,
                                    l_subroot=args.L_subfolder,
                                    mode='val')
        val_gen = get_val_examples

    else:
        dataset_test  = ImageDataset(args.data_folder, 
                                    transform=transform, 
                                    a_subroot=args.A_subfolder, 
                                    b_subroot=args.B_subfolder,
                                    l_subroot=args.L_subfolder,
                                    mode='test')


    ## Create Criterion
    # adverarial
    adv_criterion = nn.MSELoss() 
    # identity
    idn_criterion = nn.L1Loss() 
    # cycle
    if args.iv3:
        inception_model = get_inception_v3()
        cyc_criterion = partial(inception_loss, inception_model, nn.L1Loss())
    else:
        cyc_criterion = nn.L1Loss()



    ## Create Generator and Discriminator
    gen_AB = Generator(dim_A, dim_B, num_res=args.gen_res_blocks).to(args.device)
    gen_BA = Generator(dim_B, dim_A, num_res=args.gen_res_blocks).to(args.device)
    gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=args.lr, betas=(0.5, 0.999))
    disc_A = Discriminator(dim_A).to(args.device)
    disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    disc_B = Discriminator(dim_B).to(args.device)
    disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # reconstruction discriminator
    disc_L = Discriminator(dim_L).to(args.device)
    disc_L_opt = torch.optim.Adam(disc_L.parameters(), lr=args.lr, betas=(0.5, 0.999))

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
        disc_L.load_state_dict(pre_dict['disc_L'])
        disc_L_opt.load_state_dict(pre_dict['disc_L_opt'])
    else:
        if args.save:
            args.save_path += 'cycleGAN_'
            print(f'Model will be saved to {args.save_path}XXX.pth')
        gen_AB = gen_AB.apply(weights_init)
        gen_BA = gen_BA.apply(weights_init)
        disc_A = disc_A.apply(weights_init)
        disc_B = disc_B.apply(weights_init)
        disc_L = disc_L.apply(weights_init)

    # Landmarks
    if not args.landmarks:
        disc_L = None
        disc_L_opt = None

    # Train

    if args.train:
        # Tensorboard summary writer
        logdir = 'runs/' + datetime.now().strftime("%d_%m_%Y__%H_%M_%S_") \
                            + f'lr{args.lr}_wcl{args.lambda_cycle}_wrl{args.lambda_rec}/'
        train_writer = SummaryWriter(logdir + 'train')
        val_writer = SummaryWriter(logdir + 'val')

        mean_generator_loss = 0
        mean_discriminator_loss = 0
        # initialize Data Loaders
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
        dataloader_val   = DataLoader(dataset_val,   batch_size=args.batch_size, shuffle=True)
        cur_step = 0

        for epoch in range(args.epochs):
            print(f'Epoch {epoch}/{args.epochs}')
            # Dataloader returns the batches
            for real_A, real_B, landmarks_B in tqdm(dataloader_train):
                real_A = nn.functional.interpolate(real_A, size=target_shape)
                real_B = nn.functional.interpolate(real_B, size=target_shape)
                landmarks_B = nn.functional.interpolate(landmarks_B, size=target_shape)
                cur_batch_size = len(real_A)
                real_A = real_A.to(args.device)
                real_B = real_B.to(args.device)
                landmarks_B = landmarks_B.to(args.device)

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

                ## Update Reconstruction Discriminator L ##
                if args.landmarks:
                    with torch.no_grad():
                        rec_B = gen_AB(gen_BA(real_B))
                    disc_L_loss = get_disc_loss_L(real_B, rec_B, landmarks_B, disc_L, adv_criterion)
                    disc_L_loss.backward(retain_graph=True) # Update gradients
                    disc_L_opt.step() # Update optimizer

                ### Update generator ###
                gen_opt.zero_grad()
                gen_loss, fake_A, fake_B = get_gen_loss(
                    real_A, real_B, landmarks_B, 
                    gen_AB, gen_BA, disc_A, disc_B, disc_L,
                    adv_criterion, idn_criterion, cyc_criterion, 
                    args.lambda_identity, args.lambda_cycle, args.lambda_rec
                )
                gen_loss.backward() # Update gradients
                gen_opt.step() # Update optimizer

                ## Update Reconstruction Discriminator L ##
                if args.landmarks:
                    with torch.no_grad():
                        rec_B = gen_AB(gen_BA(real_B))
                    disc_L_loss = get_disc_loss_L(real_B, rec_B, landmarks_B, disc_L, adv_criterion)
                    disc_L_loss.backward(retain_graph=True) # Update gradients
                    disc_L_opt.step() # Update optimizer


                # Keep track of the average discriminator loss
                mean_discriminator_loss += disc_A_loss.item() / args.val_step
                # Keep track of the average generator loss
                mean_generator_loss += gen_loss.item() / args.val_step

                ### Tensorboard ###
                if cur_step % args.val_step == 0:
                    # Mean Losses
                    train_writer.add_scalar("Mean Generator Loss", mean_generator_loss, cur_step)
                    train_writer.add_scalar("Mean Discriminator Loss", mean_discriminator_loss, cur_step)

                    mean_generator_loss = 0
                    mean_discriminator_loss = 0

                    val_A, val_B, val_landmarks_B = next(val_gen())
                    val_A = nn.functional.interpolate(val_A, size=target_shape)
                    val_B = nn.functional.interpolate(val_B, size=target_shape)
                    val_landmarks_B = nn.functional.interpolate(val_landmarks_B, size=target_shape)
                    val_A = val_A.to(args.device)
                    val_B = val_B.to(args.device)
                    val_landmarks_B = val_landmarks_B.to(args.device)

                    # Specific Losses
                    # train
                    train_losses = get_gen_losses( real_A, real_B, landmarks_B,
                                                   gen_AB, gen_BA, 
                                                   disc_A, disc_B, disc_L,
                                                   adv_criterion, 
                                                   idn_criterion, 
                                                   cyc_criterion)
                    # val
                    val_losses   = get_gen_losses(val_A, val_B, val_landmarks_B,
                                                  gen_AB, gen_BA, 
                                                  disc_A, disc_B, disc_L, 
                                                  adv_criterion, 
                                                  idn_criterion, 
                                                  cyc_criterion)

                    if args.landmarks:
                        adv_train, idn_train, cyc_train, rec_train = train_losses
                        adv_val  , idn_val  , cyc_val  , rec_val   = val_losses
                    else:
                        adv_train, idn_train, cyc_train = train_losses
                        adv_val  , idn_val  , cyc_val   = val_losses

                    #Write
                    train_writer.add_scalar("Adversarial Loss", adv_train, cur_step)
                    train_writer.add_scalar("Identity Loss", idn_train, cur_step)
                    train_writer.add_scalar("Cycle-Consistency Loss", cyc_train, cur_step)

                    val_writer.add_scalar("Adversarial Loss", adv_val, cur_step)
                    val_writer.add_scalar("Identity Loss", idn_val, cur_step)
                    val_writer.add_scalar("Cycle-Consistency Loss", cyc_val, cur_step)
                    if args.landmarks:
                        train_writer.add_scalar("Rec-Adversarial Loss", rec_train, cur_step)
                        val_writer.add_scalar("Rec-Adversarial Loss", rec_val, cur_step)


                ## Save Images ##
                if cur_step % args.display_step == 0:
                    train_writer.add_image('Real AB', convert_tensor_images(torch.cat([real_A, real_B], dim=-1), size=(dim_A, target_shape, target_shape)), cur_step)
                    train_writer.add_image('Fake BA', convert_tensor_images(torch.cat([fake_B, fake_A], dim=-1), size=(dim_A, target_shape, target_shape)), cur_step)
                cur_step += 1


            train_writer.flush()

            ## Model Saving ##
            if args.save and epoch % args.save_epochs == 0:
                save_model()
        if args.save:        
            save_model()


if __name__ == '__main__':

    # get arguments
    args = parse_args()
    print(args)
    
    main(args)
