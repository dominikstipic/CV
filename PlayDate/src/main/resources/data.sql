-- username: johndoe(or john.doe@gmail.com), password: asd, role: ADMIN
INSERT INTO users (id, first_name, last_name, username, gender, password, email, birthday, country, city, phone, about, enabled, credentials_expired, expired, locked) VALUES (1, 'John', 'Doe','johndoe', 'MALE', '$2a$10$hJbeBbKmRjoatI0BmelmZOLhWhvsw3txwgkvHzRl.f3fm1Xa.v2bm', 'john.doe@gmail.com', CURRENT_DATE-7300, 'Croatia', 'Zagreb', '099-264-7899', 'Father of 2 from the city of Zagreb. Like playing basketball with my kids.',true, false, false, false);
-- username: johndoe2(or john.doe2@gmail.com), password: asd, role: USER
INSERT INTO users (id, first_name, last_name, username, gender, password, email, birthday, country, city, phone, about, enabled, credentials_expired, expired, locked) VALUES (2, 'Jane', 'Doe', 'janedoe', 'FEMALE', '$2a$10$hJbeBbKmRjoatI0BmelmZOLhWhvsw3txwgkvHzRl.f3fm1Xa.v2bm', 'jane.doe@gmail.com', CURRENT_DATE-7300, 'Slovenia', 'Ljubljana', '099-264-7899', 'Mother of 2 from Slovenia', true, false, false, false);

INSERT INTO roles (id, role_name) VALUES (1, 'ROLE_USER');
INSERT INTO roles (id, role_name) VALUES (2, 'ROLE_ADMIN');

INSERT INTO user_roles (user_id, role_id) SELECT u.id, r.id FROM users u, roles r WHERE u.username = 'johndoe' and r.id = 2;
INSERT INTO user_roles (user_id, role_id) SELECT u.id, r.id FROM users u, roles r WHERE u.username = 'janedoe' and r.id = 1;

INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (1, false, 'This is test post 1.', CURRENT_TIMESTAMP-1, 'TEXT', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (2, false, 'This is test post 2.', CURRENT_TIMESTAMP-1, 'DEMAND', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (3, false, 'This is test post 3.', CURRENT_TIMESTAMP-1, 'OFFER', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (4, false, 'This is test post 4.', CURRENT_TIMESTAMP-2, 'DEMAND', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (5, false, 'This is test post 5.', CURRENT_TIMESTAMP-3, 'TEXT', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (6, false, 'This is test post 6.', CURRENT_TIMESTAMP-4, 'OFFER', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (7, false, 'This is test post 7.', CURRENT_TIMESTAMP-5, 'TEXT', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (8, false, 'This is test post 8.', CURRENT_TIMESTAMP-6, 'TEXT', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (9, false, 'This is test post 9.', CURRENT_TIMESTAMP-7, 'OFFER', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (10, false, 'This is test post 10.', CURRENT_TIMESTAMP-8, 'TEXT', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (11, false, 'This is test post 11.', CURRENT_TIMESTAMP-9, 'OFFER', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (12, false, 'This is test post 12.', CURRENT_TIMESTAMP-10, 'TEXT', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (13, false, 'This is test post 13.', CURRENT_TIMESTAMP-11, 'DEMAND', 2);

INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (14, false, 'This is test post 14.', CURRENT_TIMESTAMP-7, 'TEXT', 2);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (15, false, 'This is test post 15.', CURRENT_TIMESTAMP-8, 'DEMAND', 2);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (16, false, 'This is test post 16.', CURRENT_TIMESTAMP-9, 'OFFER', 2);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (17, false, 'This is test post 17.', CURRENT_TIMESTAMP-10, 'OFFER', 2);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (18, false, 'This is test post 18.', CURRENT_TIMESTAMP-11, 'TEXT', 2);

INSERT INTO groups (group_name, description, is_private, admin_id) VALUES ('Mali Nogomet', 'Grupa za dogovaranje igranja nogometa', false, 1);
INSERT INTO groups (group_name, description, is_private, admin_id) VALUES ('Velika Košarka', 'Grupa za dogovaranje igranja košarke', false, 1);
INSERT INTO groups (group_name, description, is_private, admin_id) VALUES ('Tenis', 'Grupa za dogovaranje igranja tenisa', false, 1);

INSERT INTO group_id (id, group_id) VALUES (1, 2);
INSERT INTO group_id (id, group_id) VALUES (2, 2);
INSERT INTO group_id (id, group_id) VALUES (3, 2);
INSERT INTO group_id (id, group_id) VALUES (4, 2);
INSERT INTO group_id (id, group_id) VALUES (5, 2);
INSERT INTO group_id (id, group_id) VALUES (6, 2);
INSERT INTO group_id (id, group_id) VALUES (7, 2);
INSERT INTO group_id (id, group_id) VALUES (8, 2);

INSERT INTO user_groups (user_id, group_id) VALUES (1, 1);
INSERT INTO user_groups (user_id, group_id) VALUES (1, 2);
INSERT INTO user_groups (user_id, group_id) VALUES (2, 1);

INSERT INTO comments  (id, content, owner_id, post_id) VALUES (1, 'Test comment 1.', 1, 1);
INSERT INTO comments  (id, content, owner_id, post_id) VALUES (2, 'Test comment 2.', 2, 1);
INSERT INTO comments  (id, content, owner_id, post_id) VALUES (3, 'Test comment 3.', 1, 1);
INSERT INTO comments  (id, content, owner_id, post_id) VALUES (4, 'Test comment 4.', 2, 1);
INSERT INTO comments  (id, content, owner_id, post_id) VALUES (5, 'Test comment 5.', 1, 1);
INSERT INTO comments  (id, content, owner_id, post_id) VALUES (6, 'Test comment 6.', 2, 3);

INSERT INTO user_friends VALUES (1, 2);

INSERT INTO diaries (owner_id) VALUES (1);
INSERT INTO diaries (owner_id) VALUES (2);

INSERT INTO posts (id, closed, content, date_posted, type, diary_id, owner_id) VALUES (19, false, 'This is diary test post 1.', CURRENT_TIMESTAMP-1, 'TEXT', 1, 1);
INSERT INTO posts (id, closed, content, date_posted, type, diary_id, owner_id) VALUES (20, true, 'This is diary test post 2.', CURRENT_TIMESTAMP-1, 'DEMAND', 1, 1);
INSERT INTO posts (id, closed, content, date_posted, type, diary_id, owner_id) VALUES (21, false, 'This is diary test post 3.', CURRENT_TIMESTAMP-1, 'OFFER', 1, 1);
INSERT INTO posts (id, closed, content, date_posted, type, diary_id, owner_id) VALUES (22, false, 'This is diary test post 4.', CURRENT_TIMESTAMP-2, 'DEMAND', 1, 1);
INSERT INTO posts (id, closed, content, date_posted, type, diary_id, owner_id) VALUES (23, false, 'This is diary test post 5.', CURRENT_TIMESTAMP-3, 'TEXT', 1, 1);
INSERT INTO posts (id, closed, content, date_posted, type, diary_id, owner_id) VALUES (24, true, 'This is diary test post 6.', CURRENT_TIMESTAMP-4, 'OFFER', 2, 2);
INSERT INTO posts (id, closed, content, date_posted, type, diary_id, owner_id) VALUES (25, false, 'This is diary test post 7.', CURRENT_TIMESTAMP-5, 'TEXT', 2, 2);

INSERT INTO wishlists (owner_id) VALUES (1);
INSERT INTO wishlists (owner_id) VALUES (2);

INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (26, false, 'This is wishlist test post 1.', CURRENT_TIMESTAMP-1, 'TEXT', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (27, false, 'This is wishlist test post 2.', CURRENT_TIMESTAMP-1, 'DEMAND', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (28, true, 'This is wishlist test post 3.', CURRENT_TIMESTAMP-1, 'OFFER', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (29, false, 'This is wishlist test post 4.', CURRENT_TIMESTAMP-2, 'DEMAND', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (30, false, 'This is wishlist test post 5.', CURRENT_TIMESTAMP-3, 'TEXT', 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (31, true, 'This is wishlist test post 6.', CURRENT_TIMESTAMP-4, 'OFFER', 2);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id) VALUES (32, false, 'This is wishlist test post 7.', CURRENT_TIMESTAMP-5, 'TEXT', 2);

INSERT INTO post_wishlists (post_id, wishlist_id) VALUES (26, 1);
INSERT INTO post_wishlists (post_id, wishlist_id) VALUES (27, 1);
INSERT INTO post_wishlists (post_id, wishlist_id) VALUES (28, 1);
INSERT INTO post_wishlists (post_id, wishlist_id) VALUES (29, 1);
INSERT INTO post_wishlists (post_id, wishlist_id) VALUES (30, 1);
INSERT INTO post_wishlists (post_id, wishlist_id) VALUES (31, 2);
INSERT INTO post_wishlists (post_id, wishlist_id) VALUES (32, 2);


INSERT INTO calendars(owner_id) values (1);
INSERT INTO calendars(owner_id) values (2);

INSERT INTO calendar_entries(id, start, END, title, repeat_interval, calendar_id)
values(1,'2018-12-26 08:31:01.328' :: TIMESTAMP, '2018-12-26 09:31:01.328' :: TIMESTAMP, 'content1', 'DAILY',1 );
INSERT INTO calendar_entries(id, start, END, title, repeat_interval, calendar_id)
values(2,'2018-12-20 19:31:01.328' :: TIMESTAMP, '2018-12-20 20:31:01.328' :: TIMESTAMP, 'content2', 'DAILY',1);

INSERT INTO activity_maps(owner_id) VALUES (1);
INSERT INTO activity_maps(owner_id) VALUES (2);

INSERT INTO posts (id, closed, content, date_posted, type, owner_id, activity_map_id) VALUES (33, false, 'This is activity test post 1.', CURRENT_TIMESTAMP-1, 'TEXT', 1, 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id, activity_map_id) VALUES (34, false, 'This is activity test post 2.', CURRENT_TIMESTAMP-1, 'DEMAND', 1, 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id, activity_map_id) VALUES (35, true, 'This is activity test post 3.', CURRENT_TIMESTAMP-1, 'OFFER', 1, 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id, activity_map_id) VALUES (36, false, 'This is activity test post 4.', CURRENT_TIMESTAMP-2, 'DEMAND', 1, 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id, activity_map_id) VALUES (37, false, 'This is activity test post 5.', CURRENT_TIMESTAMP-3, 'TEXT', 1, 1);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id, activity_map_id) VALUES (38, true, 'This is activity test post 6.', CURRENT_TIMESTAMP-4, 'OFFER', 2, 2);
INSERT INTO posts (id, closed, content, date_posted, type, owner_id, activity_map_id) VALUES (39, false, 'This is activity test post 7.', CURRENT_TIMESTAMP-5, 'TEXT', 2, 2);

INSERT INTO locations ( latitude, longitude, post_id) VALUES (45, 16, 33);
INSERT INTO locations ( latitude, longitude, post_id) VALUES (44, 16, 34);
INSERT INTO locations ( latitude, longitude, post_id) VALUES (43, 16, 35);
INSERT INTO locations ( latitude, longitude, post_id) VALUES (45, 15, 36);
INSERT INTO locations ( latitude, longitude, post_id) VALUES (44, 14, 37);
INSERT INTO locations ( latitude, longitude, post_id) VALUES (44, 17, 38);
INSERT INTO locations ( latitude, longitude, post_id) VALUES (44.1, 17, 39);


INSERT INTO calendar_entries(id, start, END, title, repeat_interval, calendar_id)
values(3,'2019-01-18 08:31:01.328' :: TIMESTAMP, '2019-01-18 09:31:01.328' :: TIMESTAMP, 'content1', 'DAILY',2 );


INSERT INTO calendar_entries(id, start, END, title, repeat_interval, calendar_id)
values(4,'2019-01-17 19:31:01.328' :: TIMESTAMP, '2019-01-17 20:31:01.328' :: TIMESTAMP, 'content2', 'DAILY',2);
