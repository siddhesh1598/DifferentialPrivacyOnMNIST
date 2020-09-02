# import 
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np 


def getMNISTDataset(transform=None):
	"""
		download MNIST dataset and transform it using desired
		parameters
	"""

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))
		])

	train_data = datasets.MNIST(root='data', 
		train=True, download=True, transform=transform)
	test_data = datasets.MNIST(root='data', 
		train=False, download=True, transform=transform)

	return (train_data, test_data)


def getTeacherDataLoaders(train_data, num_teachers = 10, 
	data_size = 1000, autogen_data_size = False):
	"""
		get DataLoaders for all the teachers
	"""

	teacher_data_loaders = []

	if autogen_data_size:
    	data_size = len(train_data) // num_teachers

    for i in range(num_teachers):
        indices = list(range(i*data_size, (i+1)*data_size))
        subset = Subset(train_data, indices)
        loader = torch.utils.data.DataLoader(subset, 
        	batch_size=batch_size)
        teacher_data_loaders.append(loader)

    return teacher_data_loaders


def getStudentDataset(test_data, student_train_set_size,
	batch_size):
	"""
		get train and test data for student
	"""

	student_train_data = Subset(test_data, 
		list(range(student_train_set_size)))
	student_test_data = Subset(test_data, 
		list(range(student_train_set_size, len(test_data))))

	student_train_loader = torch.utils.data.DataLoader(
		student_train_data, batch_size=batch_size)
	student_test_loader = torch.utils.data.DataLoader(
		student_test_data, batch_size=batch_size)

	return (student_train_loader, student_test_loader)


def getStudentLabels(preds, epsilon):
	"""
		add Laplacian noise to the predicted results of the 
		teacher models
	"""

	labels = np.array([]).astype(int)

    for image_preds in np.transpose(preds):
        label_counts = np.bincount(image_preds, minlength=10)
        beta = 1 / epsilon

        for i in range(len(label_counts)):
            label_counts[i] += np.random.laplace(0, beta, 1)
        
        new_label = np.argmax(label_counts)
        labels = np.append(labels, new_label)

    return labels


def studentDataLoader(student_train_loader, labels):
	"""
		DataLoader for student training dataset
	"""

	student_iterator = iter(student_train_loader)
    
    for i, (data, _) in enumerate(student_iterator):
        student_train_label = torch.from_numpy(
        	labels[i*len(data):(i+1)*len(data)])
        yield data, student_train_label