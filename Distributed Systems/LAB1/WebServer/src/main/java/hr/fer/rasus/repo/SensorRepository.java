package hr.fer.rasus.repo;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import hr.fer.rasus.dao.SensorDescription;

@Repository
public interface SensorRepository extends JpaRepository<SensorDescription, String>{

}
