  + - - - + - + - - creation date: 15/08/23
  + - + - + created by Vladimir Baranov (Quasikot)  <br>
  + - + - + email: vsbaranov83@gmail.com  <br>      

#                     List of main modules

|    Module name    |                                    Description                                    |    Dependency    |
|:-----------------:|:---------------------------------------------------------------------------------:|:----------------:|
| screenshot.py     | Take screenshot                                                                   | PyQt5            |
| preprocess.py     | Preprocess image                                                                  | opencv           |
| classification.py | Trains CNNs to classify regions of screenshot on rectangle(or images) and text regions           | pytorch          |
| train_data_collector.py | Prepare data in form of button images or rectangular areas  for cnn training           | pytorch, PyQt5          |
| gui.py            | Shortkey and gui panel logic                                                      | PyQt5            |
| test.py           | List of tests on all subsystems of UsefulClicker                                  | All of the above |
| script_builder    | Create python script based on recording of user input events (mouse and keyboard) |                  |
|stringlen_estimation_dataset.py| Prepare data for training cnn. Load words from textbook and render it in separate file | |
|stringlen_estimation.py| CNN that can predict length of a string based on a given image | |
