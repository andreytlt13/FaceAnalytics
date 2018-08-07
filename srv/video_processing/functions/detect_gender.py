def detect_gender(faces, sess, gender, train_mode, images_pl):
    return sess.run(gender, feed_dict={images_pl: faces, train_mode: False})
