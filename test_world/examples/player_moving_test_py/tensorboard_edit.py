# keunhyung 19/1/1
# csv to tensorboard
# For editing step

import numpy as np
import csv
import tensorflow as tf

# make summary operators for tensorboard
def setup_summary():
    episode_total_reward = tf.Variable(0.)

    tf.summary.scalar('Total Reward/Episode', episode_total_reward)

    summary_vars = [episode_total_reward]
    summary_placeholders = [tf.placeholder(tf.float32) for _ in
                            range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                  range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op

data = np.loadtxt('./run_moving_test-tag-Total_Reward_Episode.csv', 
									delimiter=',', dtype='float32')

# for tensorboard
sess = tf.InteractiveSession()
summary_placeholders, update_ops, summary_op = setup_summary()

summary_writer = \
    tf.summary.FileWriter('./tmp', sess.graph)

sess.run(tf.global_variables_initializer())

for line in data:
	stats= [line[2]]
	for i in range(len(stats)):
		sess.run(update_ops[i], 
			feed_dict={summary_placeholders[i]: float(stats[i])})
	summary_str = sess.run(summary_op)
	summary_writer.add_summary(summary_str, line[1]-6000)