package com.playdate.services;

import com.playdate.models.Group;
import com.playdate.models.User;
import com.playdate.models.dtos.GroupDto;
import com.playdate.models.dtos.UserDto;
import org.apache.lucene.search.Query;
import org.hibernate.search.jpa.FullTextEntityManager;
import org.hibernate.search.query.dsl.QueryBuilder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.persistence.EntityManager;
import javax.persistence.PersistenceContext;
import java.util.ArrayList;
import java.util.List;

@Service
@Transactional
public class SearchService {

    @PersistenceContext
    private final EntityManager entityManager;

    public SearchService(EntityManager entityManager) {
        this.entityManager = entityManager;
    }

    public List<UserDto> searchUsers(String search) {
        FullTextEntityManager fullTextEntityManager =
                org.hibernate.search.jpa.Search.
                        getFullTextEntityManager(entityManager);

        QueryBuilder qb = fullTextEntityManager.getSearchFactory().buildQueryBuilder().forEntity(User.class).get();

        String query = search.toLowerCase() + "*";

        Query luceneQuery = qb
                .keyword()
                .wildcard()
                .onField("firstName")
                .boostedTo(1.5f)
                .andField("lastName")
                .matching(query)
                .createQuery();


        org.hibernate.search.jpa.FullTextQuery jpaQuery = fullTextEntityManager.createFullTextQuery(luceneQuery, User.class);

        List<UserDto> users = new ArrayList<>();
        for (User u : ((List<User>) jpaQuery.getResultList())) {
            UserDto userDto = new UserDto(u.getFirstName(), u.getLastName(), u.getUsername(), u.getEmail(),u.getId());
            users.add(userDto);
        }

        return users;
    }

    public List<GroupDto> searchGroups(String search) {
        FullTextEntityManager fullTextEntityManager =
                org.hibernate.search.jpa.Search.
                        getFullTextEntityManager(entityManager);

        QueryBuilder qb = fullTextEntityManager.getSearchFactory().buildQueryBuilder().forEntity(Group.class).get();

        String query = "*" + search.toLowerCase() + "*";
        Query luceneQuery = qb
                .keyword()
                .wildcard()
                .onField("groupName")
                .matching(query)
                .createQuery();

        org.hibernate.search.jpa.FullTextQuery jpaQuery = fullTextEntityManager.createFullTextQuery(luceneQuery, Group.class);

        List<GroupDto> groups = new ArrayList<>();
        for (Group g : ((List<Group>) jpaQuery.getResultList())) {
            GroupDto groupDto = new GroupDto(g.getId(), g.getGroupName());
            groups.add(groupDto);
        }

        return groups;
    }

    public List<UserDto> searchFriendsNotInGroup(String search, Group group, User user){
        FullTextEntityManager fullTextEntityManager =
                org.hibernate.search.jpa.Search.
                        getFullTextEntityManager(entityManager);

        QueryBuilder qb = fullTextEntityManager.getSearchFactory().buildQueryBuilder().forEntity(User.class).get();

        String query = search.toLowerCase() + "*";

        Query luceneQuery = qb
                .keyword()
                .wildcard()
                .onField("firstName")
                .boostedTo(1.5f)
                .andField("lastName")
                .matching(query)
                .createQuery();


        org.hibernate.search.jpa.FullTextQuery jpaQuery = fullTextEntityManager.createFullTextQuery(luceneQuery, User.class);

        List<UserDto> users = new ArrayList<>();
        for (User u : ((List<User>) jpaQuery.getResultList())) {
            if(user.getFriends().contains(u) && !group.getMembers().contains(u)){
                UserDto userDto = new UserDto(u.getFirstName(), u.getLastName(), u.getUsername(), u.getEmail(), u.getId());
                users.add(userDto);
            }
        }
        return users;
    }
}
