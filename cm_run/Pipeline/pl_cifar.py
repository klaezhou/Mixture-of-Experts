import torch
from data._cifar_init import data_generator 
from model.Rn_20 import ResNet20
from model.pf_vision import PF_moe


class Pipeline():
        def __init__(self,args):
            self.args=args
            
        def data_loader(self):
            data_gen=data_generator(self.args.class_index)
            x,y=data_gen.get_data()
            
            self.trainloader=x
            self.testloader=y
            
            
        def build_model(self):
            Res=ResNet20(num_classes=self.args.class_num).to(self.args.device)
            self.model=Res
            return self.model
        
        def build_gate_model(self):
            PF_model=PF_moe(self.args).to(self.args.device)
            self.PF_model=PF_model
            return self.PF_model
        
        def load_model(self,path=None):
            if path is None:
                path=self.args.path
            self.model.load_state_dict(torch.load(path))
            self.model.to(self.args.device)
            
        def load_gate_model(self):
            self.PF_model.load_state_dict(torch.load(self.args.pathpf))
            self.PF_model.to(self.args.device)
            
        def change_conv(self,kernel_size):
            self.PF_model.gating_network._set_conv(kernel_size)
        def train_model(self):
            self.model.train()
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
            criterion = torch.nn.CrossEntropyLoss()
            for epoch in range(self.args.epochs):
                for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                    inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                    targets=targets # Adjust targets to start from 0
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                if epoch % self.args.log_interval == 0:
                    print(f'Epoch {epoch}, Loss: {loss.item()}')
                    
            torch.save(self.model.state_dict(), self.args.path)
            
        def train_gate_model(self):
            self.PF_model.train()
            optimizer = torch.optim.SGD(self.PF_model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
            criterion = torch.nn.CrossEntropyLoss()
            step_count=self.args.smooth_steps
            for epoch in range(self.args.epochs):
                step_count -=1
                
                for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                    inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                    targets=targets  
                    
                    if step_count<=0 and epoch<=self.args.smooth_lb:
                        with torch.no_grad():
                            self.PF_model.gating_network.softmax_eps.copy_(torch.tensor(self.args.eps_lb
                                                                    , device=self.args.device)) 
                        step_count=self.args.smooth_steps
                        
                    optimizer.zero_grad()
                    outputs = self.PF_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                if epoch % self.args.log_interval == 0:
                    print(f'Epoch {epoch}, Loss: {loss.item()}')
                    
            torch.save(self.PF_model.state_dict(), self.args.pathpf)
        def test_PF_model(self):
            self.PF_model.eval()
                
            num_classes = 10  # 如果是 CIFAR-10
            class_names = ['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            
            correct = 0
            total = 0
            class_correct = list(0. for i in range(num_classes))
            class_total = list(0. for i in range(num_classes))

            with torch.no_grad():
                for inputs, targets in self.testloader:
                    inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                    
                    outputs = self.PF_model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # 总准确率统计
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                    # 2. 分类别统计
                    res = (predicted == targets).squeeze()
                    for i in range(len(targets)):
                        label = targets[i]
                        class_correct[label] += res[i].item()
                        class_total[label] += 1

            # 打印总准确率
            accuracy = 100 * correct / total
            print(f'Overall Test Accuracy: {accuracy:.2f}%')
            print('-' * 30)

            # 3. 打印每个类的准确率
            for i in range(num_classes):
                if class_total[i] > 0:
                    class_acc = 100 * class_correct[i] / class_total[i]
                    print(f'Accuracy of {class_names[i]:10s}: {class_acc:.2f}%')
                else:
                    print(f'Accuracy of {class_names[i]:10s}: N/A (No samples)')
        def test_model(self):
                self.model.eval()
                
                num_classes = 10  # 如果是 CIFAR-10
                class_names = ['plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                
                correct = 0
                total = 0
                class_correct = list(0. for i in range(num_classes))
                class_total = list(0. for i in range(num_classes))

                with torch.no_grad():
                    for inputs, targets in self.testloader:
                        inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                        
                        outputs = self.model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        # 总准确率统计
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()

                        # 2. 分类别统计
                        res = (predicted == targets).squeeze()
                        for i in range(len(targets)):
                            label = targets[i]
                            class_correct[label] += res[i].item()
                            class_total[label] += 1

                # 打印总准确率
                accuracy = 100 * correct / total
                print(f'Overall Test Accuracy: {accuracy:.2f}%')
                print('-' * 30)

                # 3. 打印每个类的准确率
                for i in range(num_classes):
                    if class_total[i] > 0:
                        class_acc = 100 * class_correct[i] / class_total[i]
                        print(f'Accuracy of {class_names[i]:10s}: {class_acc:.2f}%')
                    else:
                        print(f'Accuracy of {class_names[i]:10s}: N/A (No samples)')
            