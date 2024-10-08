import matplotlib.pyplot as plt

num_rounds = 201

rounds = []
for i in range(num_rounds):
    rounds.append(str(i+1))

result = {'r2': [(0, -1.08343431414028),
      	              (1, -0.811285126257911),
      	              (2, -0.47311119694610615),
      	              (3, -0.28130284473999256),
      	              (4, -0.11311006951162339),
      	              (5, 0.052305246297927876),
      	              (6, 0.12155249489779862),
      	              (7, 0.16293133088376355),
      	              (8, 0.23338852103374974),
      	              (9, 0.2686635334416375),
      	              (10, 0.3023747701360344),
      	              (11, 0.3468073969314348),
      	              (12, 0.3829125019605393),
      	              (13, 0.39515074586994636),
      	              (14, 0.4181880899124465),
      	              (15, 0.43072663479358053),
      	              (16, 0.4638079430811861),
      	              (17, 0.45873640433448515),
      	              (18, 0.44813449059331323),
      	              (19, 0.44543918741507815),
      	              (20, 0.46512975314445937),
      	              (21, 0.45778252353297877),
      	              (22, 0.4761261380178302),
      	              (23, 0.4719636475211446),
      	              (24, 0.46935709923456437),
      	              (25, 0.47562352652834516),
      	              (26, 0.4631976436392612),
      	              (27, 0.4743302644785583),
      	              (28, 0.49980273298556077),
      	              (29, 0.4845283544331437),
      	              (30, 0.4869021925931867),
      	              (31, 0.4935254047403931),
      	              (32, 0.49292060572295515),
      	              (33, 0.5157097607631274),
      	              (34, 0.52790332371851),
      	              (35, 0.5204853586681655),
      	              (36, 0.4943912797308214),
      	              (37, 0.4910764703528371),
      	              (38, 0.5024306994624594),
      	              (39, 0.5106942790092723),
      	              (40, 0.4997720572435822),
      	              (41, 0.5215059734015988),
      	              (42, 0.5333484240755286),
      	              (43, 0.5417957503430861),
      	              (44, 0.5607177945539462),
      	              (45, 0.5541490294502571),
      	              (46, 0.5350201789587747),
      	              (47, 0.5419898688221485),
      	              (48, 0.5631338950507557),
      	              (49, 0.5605928738987733),
      	              (50, 0.5559717511761821),
      	              (51, 0.5682817671524185),
      	              (52, 0.5567820194332432),
      	              (53, 0.584229982555515),
      	              (54, 0.5730070473803467),
      	              (55, 0.5773107755270221),
      	              (56, 0.5861850999383902),
      	              (57, 0.5756068560165175),
      	              (58, 0.5889161585894049),
      	              (59, 0.5879927849169063),
      	              (60, 0.5998051636523287),
      	              (61, 0.5895200081785192),
      	              (62, 0.6065759180918646),
      	              (63, 0.5924823964484047),
      	              (64, 0.5887505698059847),
      	              (65, 0.5822985414729787),
      	              (66, 0.5953186085095985),
      	              (67, 0.6061521784445727),
      	              (68, 0.5946561630186318),
      	              (69, 0.5911686457044539),
      	              (70, 0.5917051145892989),
      	              (71, 0.5824744415939624),
      	              (72, 0.5947996465535121),
      	              (73, 0.5818892500534767),
      	              (74, 0.5798175909441169),
      	              (75, 0.599766424780305),
      	              (76, 0.6082793876724322),
      	              (77, 0.6078972503915718),
      	              (78, 0.5867945266439138),
      	              (79, 0.5962191112774924),
      	              (80, 0.6026052761745875),
      	              (81, 0.6034126154833694),
      	              (82, 0.5925645044027696),
      	              (83, 0.6118438365744657),
      	              (84, 0.6151164323202689),
      	              (85, 0.6154749266052426),
      	              (86, 0.6187168055221257),
      	              (87, 0.6105451968608446),
      	              (88, 0.6177686536407555),
      	              (89, 0.6009508357689306),
      	              (90, 0.5865966469102531),
      	              (91, 0.5796533104950787),
      	              (92, 0.5976530196454244),
      	              (93, 0.572227998009402),
      	              (94, 0.5662004905975158),
      	              (95, 0.5853785916008651),
      	              (96, 0.5807462852287502),
      	              (97, 0.5942253228077876),
      	              (98, 0.5658394037880152),
      	              (99, 0.5807657124513994),
      	              (100, 0.5862619417714287),
      	              (101, 0.5606843933033812),
      	              (102, 0.5929976661027123),
      	              (103, 0.5778488821680036),
      	              (104, 0.5671901245127202),
      	              (105, 0.5443040506682806),
      	              (106, 0.5877801503623998),
      	              (107, 0.5252417413803068),
      	              (108, 0.5909305323924642),
      	              (109, 0.5480442462073849),
      	              (110, 0.5744739216406671),
      	              (111, 0.5688056784072983),
      	              (112, 0.5612131412619168),
      	              (113, 0.5686981843186003),
      	              (114, 0.5254028748400199),
      	              (115, 0.552296600674155),
      	              (116, 0.593108359132072),
      	              (117, 0.581747317094408),
      	              (118, 0.5758171395951415),
      	              (119, 0.5569969858696302),
      	              (120, 0.5782099974766546),
      	              (121, 0.6012344076617686),
      	              (122, 0.5859775878955287),
      	              (123, 0.5466471348478886),
      	              (124, 0.5289684841212119),
      	              (125, 0.5561845397373858),
      	              (126, 0.5724499137196235),
      	              (127, 0.5412021564730477),
      	              (128, 0.5339898188655932),
      	              (129, 0.5178795931033702),
      	              (130, 0.5199437789920455),
      	              (131, 0.5462052698405209),
      	              (132, 0.5424900151251224),
      	              (133, 0.5346038271199667),
      	              (134, 0.5339732463445217),
      	              (135, 0.5324429353498279),
      	              (136, 0.548555994440251),
      	              (137, 0.515073309666297),
      	              (138, 0.5353797042509802),
      	              (139, 0.566550374535663),
      	              (140, 0.5465094600075933),
      	              (141, 0.5439526425902755),
      	              (142, 0.5554312476760767),
      	              (143, 0.5332421130162438),
      	              (144, 0.5463450926691578),
      	              (145, 0.5579039742076688),
      	              (146, 0.54435898843823),
      	              (147, 0.5184201072164567),
      	              (148, 0.49872433588179577),
      	              (149, 0.4978933258667314),
      	              (150, 0.4833321586065048),
      	              (151, 0.5327973672225192),
      	              (152, 0.5306410654249878),
      	              (153, 0.5041665167269338),
      	              (154, 0.543758696772258),
      	              (155, 0.4923873668951917),
      	              (156, 0.5271893789691109),
      	              (157, 0.5222681847484743),
      	              (158, 0.5614637040272354),
      	              (159, 0.5335849434767724),
      	              (160, 0.5447352076742895),
      	              (161, 0.5415927535627083),
      	              (162, 0.5299602762970101),
      	              (163, 0.5220269912069868),
      	              (164, 0.5029796914008664),
      	              (165, 0.5602549770190043),
      	              (166, 0.5479195389660285),
      	              (167, 0.5198480852355123),
      	              (168, 0.525501064426753),
      	              (169, 0.5037938197149823),
      	              (170, 0.5057287686528498),
      	              (171, 0.48928677928302566),
      	              (172, 0.5133642981813731),
      	              (173, 0.5210403041036122),
      	              (174, 0.5081144608203069),
      	              (175, 0.5303663943584112),
      	              (176, 0.5133516402138425),
      	              (177, 0.5290758625880827),
      	              (178, 0.5252876634716779),
      	              (179, 0.4970101752621455),
      	              (180, 0.5174651142673063),
      	              (181, 0.5131218835364314),
      	              (182, 0.5209488659468906),
      	              (183, 0.5127169538106573),
      	              (184, 0.5326935352903284),
      	              (185, 0.5456127902613228),
      	              (186, 0.5225204578317411),
      	              (187, 0.5610860703926883),
      	              (188, 0.5617044440049446),
      	              (189, 0.5574317778990103),
      	              (190, 0.5416867656006592),
      	              (191, 0.5539985753779508),
      	              (192, 0.5340140750987123),
      	              (193, 0.5542822536933691),
      	              (194, 0.5846011445553951),
      	              (195, 0.5574245141015941),
      	              (196, 0.5614312720696895),
      	              (197, 0.5392765253714603),
      	              (198, 0.5027712967673359),
      	              (199, 0.5267781349116838),
      	              (200, 0.5220801309311279)]}





global_r2 = []

global_mse = []

for i in range(len(result['r2'])):
      global_r2.append(result['r2'][i][1])
      

plt.figure(figsize=(20,6))
      
plt.plot(global_r2)

plt.xlabel('Round')

plt.ylabel('R2-Score')

plt.title('R2-score of the global model')

plt.grid()

plt.show()    
