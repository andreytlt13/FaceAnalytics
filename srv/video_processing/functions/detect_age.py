def detect_age(faces, sess, age, train_mode, images_pl):
    return sess.run(age, feed_dict={images_pl: faces, train_mode: False})
