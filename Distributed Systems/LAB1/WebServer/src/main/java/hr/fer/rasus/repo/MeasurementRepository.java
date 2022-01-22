package hr.fer.rasus.repo;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import hr.fer.rasus.dao.Measurement;

@Repository
public interface MeasurementRepository extends JpaRepository<Measurement, Long>{

}
