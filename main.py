# import 
import torch
import torch.optim as optim
from syft.frameworks.torch.dp import pate

from utils import *
from model import Network


# define parameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_TEACHERS = 10
TEACHER_DATASET_SIZE = 1000
STUDENT_DATASET_SIZE = 1000
EPSILON = 0.2
DELTA = 1e-5

# allocate GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get MNIST dataset
(train_data, test_data) = getMNISTDataset()

# get DataLoader for all teachers
teacher_data_loaders = getTeacherDataLoaders(train_data, 
	num_teachers=NUM_TEACHERS, data_size=TEACHER_DATASET_SIZE)

# get student train and test DataLoader
(student_train_loader, student_test_loader) = getStudentDataset(
	test_data, student_train_set_size = STUDENT_DATASET_SIZE,
	batch_size)


##################################################
#			Train teacher models 				 #
##################################################

# function to train a teacher model
def trainTeacher(model, train_loader, epochs, device=device):

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.to(device)
    running_loss = 0
    
    for e in range(epochs):

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        if (e+1)%10 == 0:
            print("Train loss for epoch {}: {:.4f}".format(e+1, running_loss/len(trainloader)))
        running_loss = 0

# train all the teachers
teacher_models = []

for teach in range(NUM_TEACHERS):
    print("Training teacher: {}".format(t+1))
    model = Network()
    train(model, teacher_data_loaders[t], epochs=EPOCHS)
    teacher_models.append(model)


##################################################
#		Predict labels for student data			 #
##################################################

# function to predict labels for a teacher model
def predict(model, dataloader, device=device):

    outputs = torch.zeros(0, dtype=torch.long).to(device)
    model.to(device)
    model.eval()

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        ps = torch.argmax(torch.exp(output), dim=1)
        outputs = torch.cat((outputs, ps))

    return outputs

preds = torch.torch.zeros((len(teacher_models), STUDENT_DATASET_SIZE), 
	dtype=torch.long)

for i, model in enumerate(teacher_models):
	results = predict(model, data_loader)
	preds[i] = results

# add noise to the predicted labels
student_labels = getStudentLabels(preds, epsilon=EPSILON)


##################################################
#				PATE Analysis 				 	 #
##################################################

# perform PATE analysis
data_dep_eps, data_ind_eps = pate.perform_analysis(
	teacher_preds=preds, indices=student_labels, 
	noise_eps=EPSILON, delta=DELTA)
print("Data Independent Epsilon:", data_ind_eps)
print("Data Dependent Epsilon:", data_dep_eps)


##################################################
#	Train student model on modified labels		 #
##################################################

# train student model on the student train data and the 
# calculated labels
student_model = Network()
criterion = nn.NLLLoss()
optimizer = optim.Adam(student_model.parameters(), 
	lr=LEARNING_RATE)
student_model.to(device)

steps = 0
running_loss = 0

for e in range(EPOCHS):
    train_loader = studentDataLoader(student_train_loader, 
    	student_labels)
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        steps += 1
        optimizer.zero_grad()

        output = student_model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % 50 == 0:
            test_loss = 0
            accuracy = 0

            with torch.no_grad():
                for images, labels in student_test_loader:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = student_model(images)
                    test_loss += criterion(log_ps, labels).item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

    print("Epoch: {}/{}".format(e+1, epochs))
    print("Training Loss: {:.3f} ".format(running_loss/len(student_train_loader)), 
            "Test Loss: {:.3f} ".format(test_loss/len(student_test_loader)), 
            "Test Accuracy: {:.3f}".format(accuracy/len(student_test_loader)))
    running_loss = 0
