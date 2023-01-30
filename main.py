from utils import config
from data.process.data_loader import prepare_data_seq
from models.CAB.model import EmoActCvae
from models.CAB.model_Wo_act import EmoActCvae_woAct
from models.CAB.model_Wo_cog import EmoActCvae_woCog
from models.CAB.model_Wo_emo import EmoActCvae_woEmo
from models.CAB.others_layer import count_parameters, set_seed, evaluate
from utils.config import args
from copy import deepcopy
from tqdm import tqdm
import logging
import math
import torch
import os
from tensorboardX import SummaryWriter

set_seed(42)
tra_data_loader, val_data_loader, tst_data_loader, vocab, emotion_number, act_number = prepare_data_seq(batch_size=config.batch_size)

if args.model == 'CAB':
    model = EmoActCvae(vocab,emo_number=emotion_number, act_number = act_number, model_file_path=config.save_path_pretrained, load_optim=config.load_optim)
elif args.model == 'Wo_cog':
    model = EmoActCvae_woCog(vocab,emo_number=emotion_number, act_number = act_number, model_file_path=config.save_path_pretrained, load_optim=config.load_optim)
elif args.model == 'Wo_emo':
    model = EmoActCvae_woEmo(vocab,emo_number=emotion_number, act_number = act_number, model_file_path=config.save_path_pretrained, load_optim=config.load_optim)
else:
    model = EmoActCvae_woAct(vocab,emo_number=emotion_number, act_number = act_number, model_file_path=config.save_path_pretrained, load_optim=config.load_optim)

print('Using model:{}'.format(args.model))
print('Training parameters:',count_parameters(model))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
print_file = args.print_file

if args.test is False:
    try:
        model = model.train()
        best_ppl = 1000
        writer = SummaryWriter(log_dir=config.save_path)
        weights_best = deepcopy(model.state_dict())

        val_print_path = os.path.join(print_file, 'valid/')
        if os.path.exists(val_print_path) is False: os.makedirs(val_print_path)
        val_result = open(os.path.join(val_print_path, 'result.txt'), 'w', encoding='utf-8')

        for epoch in range(config.num_epochs):
            total_loss, total_loss_rec, total_emo_s_loss, total_emo_l_loss, total_a_loss, total_kld, total_bow, total_elbo = 0,0,0,0,0,0,0,0
            for batch, inputs in tqdm(enumerate(tra_data_loader),total= len(tra_data_loader)):
                n_iter = epoch * len(tra_data_loader) + batch
                loss, loss_rec, emo_s_loss, emo_l_loss, a_loss, kld, bow, elbo = model.train_one_batch(inputs, n_iter)
                total_loss += loss
                total_loss_rec += loss_rec
                total_emo_s_loss += emo_s_loss
                total_emo_l_loss += emo_l_loss
                total_a_loss += a_loss
                total_kld += kld
                total_bow += bow
                total_elbo += elbo
                if (batch%500 == 0):
                    torch.cuda.empty_cache()
                    print('\nEpoch {} Batch {} Loss {:.4f} Rec_loss {:.4f} PPL {:.4f} Emo_s_loss {:.4f} Emo_l_loss {:.4f} A_loss {:.4f} Kld {:.4f} Bow {:.4f} Elbo {:.4f}'.format(
                    epoch + 1, batch, total_loss/(batch+1), total_loss_rec/(batch+1), math.exp(total_loss_rec/(batch+1)), total_emo_s_loss/(batch+1), total_emo_l_loss/(batch+1), total_a_loss/(batch+1), total_kld/(batch+1), total_bow/(batch+1), total_elbo/(batch+1)))
                writer.add_scalars('loss', {'loss_train': loss}, epoch)
                writer.add_scalars('loss_rec', {'loss_rec_train': loss_rec}, epoch)
                writer.add_scalars('ppl', {'ppl_train': math.exp(total_loss_rec/(batch+1))}, epoch)
                writer.add_scalars('emo_s_loss', {'emo_loss_train': emo_s_loss}, epoch)
                writer.add_scalars('emo_l_loss', {'emo_loss_train': emo_l_loss}, epoch)
                writer.add_scalars('a_loss', {'a_loss_train': a_loss}, epoch)
                writer.add_scalars('kld', {'kld_train': kld}, epoch)
                writer.add_scalars('bow', {'bow_train': bow}, epoch)
                writer.add_scalars('elbo', {'elbo_train': elbo}, epoch)

            model = model.eval()
            with torch.no_grad():
                loss_val, ppl_val, elbo_val, kld_val, bow_val, act_loss_val, s_acc, l_acc, act_acc = evaluate(model, val_data_loader, ty="val", max_dec_step=50, print_file = print_file, val_result = val_result)
                writer.add_scalars('loss', {'loss_val': loss_val}, epoch)
                writer.add_scalars('ppl', {'ppl_val': ppl_val}, epoch)
                writer.add_scalars('kld', {'kld_val': kld_val}, epoch)
                writer.add_scalars('bow', {'bow_val': bow_val}, epoch)
                writer.add_scalars('elbo', {'elbo_val': elbo_val}, epoch)
                writer.add_scalars('s_acc', {'s_acc_val': s_acc}, epoch)
                writer.add_scalars('l_acc', {'l_acc_val': l_acc}, epoch)
                writer.add_scalars('act_acc', {'act_acc_val': act_acc}, epoch)
                print('valid.....................')
                print('Epoch {} Loss {:.4f} PPL {:.4f} Elbo {:.4f} Actloss {:.4f} S_acc {:.4f} L_acc {:.4f} Act_acc {:.4f}'.format(
                    epoch + 1, loss_val, ppl_val, elbo_val, act_loss_val, s_acc, l_acc, act_acc))
                model = model.train()
                if best_ppl > ppl_val and epoch > 5:
                    best_ppl = ppl_val
                    model.save_model(model, epoch+1, loss_val, ppl_val, kld_val, bow_val, s_acc, l_acc, act_acc, model_name = args.model)
                    weights_best = deepcopy(model.state_dict())
                    print('Saving the best checkpoint in {} and the path is {}'.format(epoch+1, os.path.join(config.save_path_pretrained,
                                       '{}_{}_{:.4f}'.format(args.model, epoch+1, ppl_val))))
                    print('Saving the best checkpoint in {} and the path is {}'.format(epoch+1, os.path.join(config.save_path_pretrained,
                                       '{}_{}_{:.4f}'.format(args.model, epoch+1, ppl_val))), file=val_result)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    ## SAVE THE BEST
    torch.save({'model': weights_best,
                # 'result': [loss_val, ppl_val, elbo_val, kld_val, bow_val, s_acc, l_acc, act_acc] },
               },os.path.join(args.save_path_, args.model+'_best.tar'))
    print('Saving the best checkpoint in {}'.format(os.path.join(args.save_path_, args.model+'_best.tar')))

    ## TESTING
    if args.train_then_test:
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        model.eval()
        with torch.no_grad():
            loss_test, ppl_test, elbo_test, kld_test, bow_test, act_loss_test, s_acc_test, l_acc_test, act_acc_test = evaluate(model,
                                                                                                                tst_data_loader,
                                                                                                                ty='test',
                                                                                                                max_dec_step=50,
                                                                                                                print_file=print_file,
                                                                                                                val_result = val_result)
    else:
        loss_test, ppl_test, elbo_test, kld_test, bow_test, act_loss_test, s_acc_test, l_acc_test, act_acc_test = 0, 0, 0, 0, 0, 0, 0, 0, 0

else:#test
    print('Testing……')
    model = model.eval()
    # checkpoint = torch.load(os.path.join(args.save_path_pretrained, 'Wo_emo_11.0000_39.0669'),
    #                         map_location=lambda storage, location: storage)

    checkpoint = torch.load(os.path.join(args.save_path_, args.model+'_best.tar'),
                            map_location=lambda storage, location: storage)
    weights_best = checkpoint['model']
    model.load_state_dict({name: weights_best[name] for name in weights_best})
    model.eval()
    loss_test, ppl_test, elbo_test, kld_test, bow_test, act_loss_test, s_acc_test, l_acc_test, act_acc_test = evaluate(model,
                                                                                                        tst_data_loader,
                                                                                                        ty='test',
                                                                                                        max_dec_step=50,
                                                                                                        print_file=print_file)

    file_summary = config.save_path+"summary.txt"
    with open(file_summary, 'w') as the_file:
        the_file.write("TEST\tLoss\tPPL\tKLD\tELBO\tsacc\tlacc\tactacc\n")
        the_file.write("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format("test",loss_test,ppl_test,kld_test,elbo_test,s_acc_test, l_acc_test, act_acc_test))

