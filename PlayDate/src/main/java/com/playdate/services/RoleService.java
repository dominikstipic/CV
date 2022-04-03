package com.playdate.services;

import com.playdate.models.Role;
import com.playdate.repositories.RoleRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
public class RoleService {

    private final RoleRepository roleRepository;

    @Value("${com.playdate.role.prefix}")
    private String rolePrefix = "ROLE_";


    public RoleService(RoleRepository roleRepository) {
        this.roleRepository = roleRepository;
    }

    public Role findRoleByRoleName(String roleName) {
        return roleRepository.findRoleByRoleName(rolePrefix+roleName.toUpperCase());
    }
}
