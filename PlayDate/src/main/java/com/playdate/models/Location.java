package com.playdate.models;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;

import javax.persistence.*;
import javax.validation.constraints.NotNull;

@Entity
@Table(name = "locations")
@Data
@EqualsAndHashCode(of = "id")
public class Location {

    @Id
    private Long id;

    private String location;

    @NotNull
    private Double longitude;

    @NotNull
    private Double latitude;

    @OneToOne(fetch = FetchType.LAZY)
    @MapsId
    @ToString.Exclude
    private Post post;
}
