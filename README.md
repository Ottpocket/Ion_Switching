# Ion_Switching

The University of Liverpool- Ion Switching competition tested competitors whether they could predict the number of ion channels open every ten-thousandth of a second given electrophysiological signal data for that time stamp.   Having a model that could successfully predict that would enable researchers to better understand numerous diseases and more clearly see what happens inside cells.  I was given 500 seconds of time to train the model and 200 seconds to test the model.  For my solution, I adapted code by  English Ions and Co.  I added a multitask penalty, LSTMs, and more features.  The multitask penalty had the greatest impact on accuracy.  Specifically, the multitask penalty added n additional heads to the network, making the network estimate n other time stamps i positions away from the current one.  This helped reduce overfitting in the network, leading to a much better score.  

The data given by the University of Liverpool consisted of a CSV file.  Each row was a ten thousandth of a second snapshot which included the electrophysiological signal and the number of open channels.  

<img src="Ion_Graph.PNG" height = 400>

As can be seen, the number of open channels strongly correlates with the fluctuations in the signal.    
