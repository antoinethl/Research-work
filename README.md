# Hybrid Method for Local Consumption Forecasting


The power grid becomes a smart grid [[1]](#1). In fact, from the first electric transmission line installed by Thomas Edison in 1882 to nowadays super-large electric grids, power systems have evolved with the advance and progress  of human industrial civilization. 

The power grid has evolved in how electricity is generated and distributed. Many power stations and renewable energy's plants are built in a trellis to deliver energy across a country. 

However, the power grid has not only evolved in its structure but also in the inclusion of prosumers, i.e.consumers with sources of production, in the management of supply and demand. The power grid includes the consumer's behaviors to regulate the energy flows and prices.

To optimize and to regulate energy flows and price, the electric system requires the knowledge of the future consumption in short term. This means knowing the global consumption and any local consumption, i.e. scaling from a country to a device, to be able to forecast their future. Estimating energy consumption is one of the critical challenges of our time and yet, the consideration for this field is still increasing. 

Besides the energy flows optimization, predicting energy consumption allows us to acquire a more thorough understanding of our modes of consumption.

Many of estimating energy studies have focused on the buildings, as it accounts for 40% of the total energy consumption according to the International Energy Agency[[2]](#2). The energy consumption is estimated for the next minutes, hours or sometimes, days.

Several companies have constituted their business model on the local energy's forecasting, thus they focus on local consumers. Those companies provide various sensors and smart meters to be able to predict the local consumption. However, the consumers don't have the property of their data nor the decision-making ability for any demand-response programs. Given this state of affairs, it is crucial to provide a free and embedded program to the consumers who want to manage their energy consumption and energy cost. This program only needs devices consumption to formulate the forecast.

Those companies predominantly focus on large to medium consumers like other companies, industries or shopping buildings. Whereas peaks and gaps in the national consumption's curve are principally due to private use, by small/local consumers according to the International Energy Agency. Hence, a massive part of the problem isn't considered.

To be competent and able to reduce losses and deliver energy by precisely identifying low consumption peaks and high consumption peaks, the power grid must adapt supply to demand through Demand-Response programs [[3]](#3).
It'll be beneficial for decision making and planning for the future democratization of renewable energy and the decentralization of the means of energy production.

Energy consumption forecasting mostly focus on predicting the consumption of a whole building, or a set of buildings.
%
Predicting the whole consumption of a building may be a source of errors. This whole consumption regroups multiple devices, and the aggregated sum can be difficult to predict since it's complex for models to distinguish between numerous devices [[4]](#4).

To reduce the complexity of this problem, we propose in this paper a per-appliance prediction rather than one total prediction.
Indeed, predicting each appliance and aggregating the results displays better performances than directly predicting the entire consumption of the building.
By applying an estimation to each device, it will be more elementary for data-driven models to distinguish the different consumption phases of each appliance.

The key contribution of our study is to provide a forecasting method usable by small consumers to allow them to produce prediction about their consumption. It is valuable for the providers to route the electricity, to expect the production and to plan their consumption knowing the price of energy. The direct application is about Demand-Response, since both providers and consumers are aware of their future.

Our approach remains a general method which adapts itself to any device. The proposed method is a hybrid method with multiple deep learning models. Models importance in the final prediction will be determined using the best suited model for a device's consumption curve. With this, some models will receive more weight in a function (called weight function) depending on their performance at a given time. The weight function is greatly inspired from Q-learning. A weight update function ensures that the most efficient models will acquire higher weight for the final estimation. To validate our approach, we tested it on the ENERTALK dataset [[5]](#5), which contains per-devices measurements from 22 houses in South Korea.


## References

<a id="1">[1]</a> 
M. Amin, “The smart-grid solution”, Nature, vol. 499, no. 7457, pp. 145–147, 2013.

<a id="2">[2]</a> 
https://www.iea.org/

<a id="3">[3]</a>
M. Shakeri, M. Shayestegan, H. Abunima, S. S. Reza, M. Akhtaruz-zaman, A. Alamoud, K. Sopian, and N. Amin, “An intelligent systemarchitecture in home energy management systems (hems) for efficient demand response in smart grid”, Energy and Buildings, vol. 138, pp. 154–164, 2017.

<a id="4">[4]</a>
K. Amasyali and N. M. El-Gohary, “A review of data-driven building energy consumption prediction studies”, Renewable and Sustainable Energy Reviews, vol. 81, pp. 1192–1205, 2018.

<a id="5">[5]</a>
https://github.com/ch-shin/ENERTALK-dataset
C. Shin, E. Lee, J. Han, J. Yim, W. Rhee, and H. Lee, “The enertalk dataset, 15hz electricity consumption data from 22 houses in korea”, Scientific data, vol. 6, no. 1, pp. 1–13, 2019.