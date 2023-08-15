| # | Test name | Description |
|---|:---:|:---:|
|  | Screenshot Test | take a screenshot and save it to screenshot.png using PyQt5 API |
|  | Button Detector Test | Detects buttons on screenshot of current screen and draw colored semi-transparent green rectangles with numbers on the detected buttons |
|  | ClickRect Test | Perform mouse click inside rectangular area |
|  | ClickImageHist Test | Find image on a screenshot using histogram analisys and click on that image. Use a string of numbers encoding the histogram as an input. |
|  | ClickImageHash Test |  |
|  | ClickText Test |  |
|  | ClickCoordinatest Test | Performs mouse click in the given coordinates X,Y |
|  | EncodeImageAsHistograms Test | Output is a binary file encoded.png |
|  | FindImageByHistogram Test | Output is a binary file encoded.png |
|  | EncodeImageAsSHA256Hash Test | Output is a hash string |
|  | FindImageByHash Test | Find image on a screenshot using hash string as an input |
|  | FindText Test | Find a text on the screenshot using character recognition (CNN). |
|  | FindButtonWithText Test | Find a button with text ons the screenshot |
|  | CNNClassifier1 Test | This test performs processing of screenshot data in a neural network that classifies screen regions on a text and image areas. |
|  |  | Draws green and red contoures on the screenshot. The color of a contour depends on class of the region. |
|  |  | Text regions highlighed in green. Rectangle areas highlighted in red. Undetected regions is not highlighted at all. |
|  |  |  |
|  | CNNCharacterRecognition Test | This is a complex test that checks how whole optical character recognition subsystem works |
|  | Type Test | Useful clicker focus on the search box of the browser and type some predefined text. Make shure that you can adjust the delay parameter |
|  | Notepad test 1 | 1. Launch Notepad<br>2. Type some text<br>3. Change a font of a text through a menu<br> |
|  | Notepad test 2 | 1. Launch Notepad<br>2. Type some text<br>3. Change a font of a text using hotkeys<br>4. Save  |
|  | Twitter test 1 | 1. Launch browser on twitter.com<br>2. Type text of a twit<br>3. Push the "send" button |
