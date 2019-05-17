from detector.config import Config
from detector.features import ImageProcess
from detector.loader import ImagesLoader
from detector.model import SiameseNetworkModel
from detector.utils import plot_history

if __name__ == "__main__":
    loader = ImagesLoader(file_ext="*.jpg")

    df = loader.prepare_dataset(balance_targets=True)

    ipo = ImageProcess(data=df)

    train_df, val_df = ipo.train_test_split()

    x_left_train, x_right_train, targs_train = ipo.get_augmented_images_arrays(train_data=train_df,
                                                                               n_new=5)

    x_left_val, x_right_val, targs_val = ipo.get_validation_images_arrays(val_data=val_df)

    snm = SiameseNetworkModel()

    # (Config.resize_width, Config.resize_height, 3)
    snm.build_model_2(img_dimension=Config.img_width)

    hist = snm.fit_model(x_left=x_left_train,
                         x_right=x_right_train,
                         y=targs_train,
                         x_val_left=x_left_val,
                         x_val_right=x_right_val,
                         y_val=targs_val,
                         e=Config.train_number_epochs,
                         add_callbacks=True)

    plot_history(hist)
