##########
train, test = split(data)
model = fit(train.X, train.y)
predictions = model.predict(test.X)
skill = compare(test.y, predictions)

scores = list()
for i in k:
	train, test = split_old(data, i)
	model = fit(train.X, train.y)
	predictions = model.predict(test.X)
	skill = compare(test.y, predictions)
	scores.append(skill)

mean_skill = sum(scores) / count(scores)

seed(1)
scores = list()
for i in k:
	train, test = split_old(data, i)
	model = fit(train.X, train.y)
	predictions = model.predict(test.X)
	skill = compare(test.y, predictions)
	scores.append(skill)

scores = list()
for i in repeats:
	run_scores = list()
	for j in k:
		train, test = split_old(data, j)
		model = fit(train.X, train.y)
		predictions = model.predict(test.X)
		skill = compare(test.y, predictions)
		run_scores.append(skill)
	scores.append(mean(run_scores))

standard_error = standard_deviation / sqrt(count(scores))

interval = standard_error * 1.96
lower_interval = mean_skill - interval
upper_interval = mean_skill + interval

https://machinelearningmastery.com/evaluate-skill-deep-learning-models/