from ml_model.train import train_model

def deploy():
    model = train_model()
    print('Model trained successfully')

if __name__ == '__main__':
    deploy()