from typing import Callable
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch import device, Tensor, tensor, no_grad, save
from tqdm import tqdm


class classificationTrainer():
    '''
    
    :param alt_loss: dwlejfwoie
    '''
    def __init__(self,model:Module,
                 train_loader:DataLoader,
                 test_loader:DataLoader,
                 optimizer:Optimizer,
                 loss:Callable,
                 alt_loss:Callable=None,
                 
                 device:device=device('cpu'),
                 dtype=float,
                 vocal:bool=True
                 ):
        '''
        Class to automatically train, test and validate PyTorch models with used for classification tasks.
        '''
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        
        self.device = device
        self.dtype = dtype
        self.vocal = vocal

        self.alt_loss = alt_loss

        self.model = self.model.to(device=device)

    def _train_epoch_loop(self):
        self.model.train()
        losses = []
        accs = []
        if self.alt_loss:
            alt_losses = []
        for (X,y) in tqdm(self.train_loader) if self.vocal else self.train_loader:
            #reset optimizer
            self.optimizer.zero_grad()
            
            #cast inputs to device & dtype
            X:Tensor = X.to(device=self.device,dtype=self.dtype)
            y:Tensor = y.to(device=self.device,dtype=self.dtype)
            
            #get predictions from model
            preds:Tensor = self.model(X)
            
            #calculate loss
            batch_loss:Tensor = self.loss(preds,y)
            losses.append(batch_loss.item())
            
            #gradient and optimization
            batch_loss.backward()
            self.optimizer.step()

            #accuracy
            probs = preds.softmax(1)
            most_probable = probs.argmax(1)
            batch_acc = ((most_probable==y).sum())/y.size(0) #correct / samples
            accs.append(batch_acc.item())
            #other loss function
            if self.alt_loss:
                alt_loss:Tensor = self.alt_loss(preds,y)
                alt_losses.append(alt_loss.item())

        mean_loss = tensor(losses).mean()
        mean_acc = tensor(accs).mean()

        if self.alt_loss:
            mean_alt_loss = tensor(alt_losses).mean()
            return mean_loss,mean_acc, mean_alt_loss
        else:
            return mean_loss,mean_acc
        

    def _test_epoch_loop(self):
        '''
        Run test loop - no training
        '''
        losses = []
        accs = []
        if self.alt_loss:
            alt_losses = []
        with no_grad(): #no gradients
            self.model.eval() #model in eval mode
            for (X,y) in tqdm(self.test_loader) if self.vocal else self.test_loader:
                X:Tensor = X.to(device=self.device,dtype=self.dtype)
                y:Tensor = y.to(device=self.device,dtype=self.dtype)
                
                preds:Tensor = self.model(X)
                batch_loss:Tensor = self.loss(preds,y)
                losses.append(batch_loss.item())

                probs = preds.softmax(1)
                most_probable = probs.argmax(1)
                batch_acc = ((most_probable==y).sum())/y.size(0) #correct / samples
                accs.append(batch_acc.item())

                if self.alt_loss:
                    alt_loss:Tensor = self.alt_loss(preds,y)
                    alt_losses.append(alt_loss.item())

            mean_loss = tensor(losses).mean()
            mean_acc = tensor(accs).mean()

            if self.alt_loss:
                mean_alt_loss = tensor(alt_losses).mean()
                return mean_loss,mean_acc, mean_alt_loss
            else:
                return mean_loss,mean_acc
        
    def train(self,epochs=None,save_pth:str=None):
        results = []
        for epoch in range(1, epochs + 1):
            if self.vocal:
                print(f'-------------------- Train Epoch {epoch} --------------------')
            
            # Run training loop and get the corresponding losses
            if self.alt_loss:
                mean_loss, mean_acc, mean_alt_loss = self._train_epoch_loop()
                epoch_result = {
                    "epoch": epoch,
                    "mean_loss": mean_loss,
                    "mean_acc": mean_acc,
                    f"mean_{self.alt_loss.__name__}": mean_alt_loss,
                }
            else:
                mean_loss, mean_acc = self._train_epoch_loop()
                epoch_result = {
                    "epoch": epoch,
                    "mean_loss": mean_loss,
                    "mean_acc": mean_acc,
                }
            
            results.append(epoch_result)

            # Print results if verbose
            if self.vocal:
                print("\n".join(f"{key}: {value}" for key, value in epoch_result.items() if key != "epoch"))
        
        if save_pth:
            save(self.model.state_dict(), save_pth)

        # Return the collected results
        return results

    def test(self,epochs=None,save_pth:str=None):
        results = []
        for epoch in range(1, epochs + 1):
            if self.vocal:
                print(f'-------------------- Test Epoch {epoch} --------------------')
            
            # Run test loop and get the corresponding losses
            if self.alt_loss:
                mean_loss, mean_acc, mean_alt_loss = self._test_epoch_loop()
                epoch_result = {
                    "epoch": epoch,
                    "mean_loss": mean_loss,
                    "mean_acc": mean_acc,
                    f"mean_{self.alt_loss.__name__}": mean_alt_loss,
                }
            else:
                mean_loss, mean_acc = self._test_epoch_loop()
                epoch_result = {
                    "epoch": epoch,
                    "mean_loss": mean_loss,
                    "mean_acc": mean_acc,
                }
            
            results.append(epoch_result)

            # Print results if verbose
            if self.vocal:
                print("\n".join(f"{key}: {value}" for key, value in epoch_result.items() if key != "epoch"))
        
        if save_pth:
            save(self.model.state_dict(), save_pth)

        # Return the collected results
        return results
        
    def train_test(self,epochs,save_pth:str=None):
        train_results = []
        test_results = []
        for epoch in range(1, epochs + 1):
            # -------------------- Train --------------------
            if self.vocal:
                print(f'-------------------- Train Epoch {epoch} --------------------')
            
            # Run training loop and get the corresponding losses
            if self.alt_loss:
                train_mean_loss, train_mean_acc, train_mean_alt_loss = self._train_epoch_loop()
                train_epoch_result = {
                    "epoch": epoch,
                    "mean_loss": train_mean_loss,
                    "mean_acc": train_mean_acc,
                    f"mean_{self.alt_loss.__name__}": train_mean_alt_loss,
                }
            else:
                train_mean_loss, train_mean_acc = self._train_epoch_loop()
                train_epoch_result = {
                    "epoch": epoch,
                    "mean_loss": train_mean_loss,
                    "mean_acc": train_mean_acc,
                }
            train_results.append(train_epoch_result)
            # Print results if verbose
            if self.vocal:
                print("\n".join(f"{key}: {value}" for key, value in train_epoch_result.items() if key != "epoch"))
            
            # -------------------- Test --------------------
            if self.vocal:
                print(f'-------------------- Test Epoch {epoch} --------------------')
            
            # Run training loop and get the corresponding losses
            if self.alt_loss:
                test_mean_loss, test_mean_acc, test_mean_alt_loss = self._test_epoch_loop()
                test_epoch_result = {
                    "epoch": epoch,
                    "mean_loss": test_mean_loss,
                    "mean_acc": test_mean_acc,
                    f"mean_{self.alt_loss.__name__}": test_mean_alt_loss,
                }
            else:
                test_mean_loss, test_mean_acc = self._test_epoch_loop()
                test_epoch_result = {
                    "epoch": epoch,
                    "mean_loss": test_mean_loss,
                    "mean_acc": test_mean_acc,
                }
            
            test_results.append(test_epoch_result)

            # Print results if verbose
            if self.vocal:
                print("\n".join(f"{key}: {value}" for key, value in test_epoch_result.items() if key != "epoch"))
        
        # Return the collected results
        if save_pth:
            save(self.model.state_dict(), save_pth)

        return train_results, test_results
        

        

    
