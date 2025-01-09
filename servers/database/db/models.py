
__all__ = ["Dataset", "Image", "ImageInfo", "User"]


import datetime, traceback, os

from sqlalchemy import create_engine, Column, Boolean, Float, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base


Base = declarative_base()


class Fold(Base):
    __tablename__ = "fold"



class Dataset(Base):
    __tablename__ = "dataset"

    id          = Column(Integer, primary_key = True)
    name        = Column(String, unique=True)
    dataset_id  = Column(String, unique=True)
    public      = Column(Boolean, default=False)
    synthetic   = Column(Boolean, default=False)
    last_update = Column(DateTime)
    comment     = Column(String)

    # Foreign  
    images      = relationship("Image", order_by="Image.id", back_populates="dataset")



class Image(Base):
    __tablename__ = "image"

    id               = Column(Integer, primary_key = True)
    image_id         = Column(String, unique=True)
    image_hash       = Column(String)
    date_acquisition = Column(Datetime)
    date_insertion   = Column(DateTime)
    project_id       = Column(String)
    original_image   = Column(Boolean)

    data             = Column()

    # Foreign
    dataset    = relationship("Dataset", back_populates="images")
    datasetid  = Column(Integer, ForeignKey('dataset.id'))
  


class ImageInfo(Base):

    __tablename___ = "image_info"
    id             = Column(Integer, primary_key = True)
    version        = Column(Integer, default=0)
    imageinfo_id   = Column(String, unique=True)
    image_id       = Column(String)
    has_tb         = Column(Boolean)
    gender         = Column(Char)
    age            = Column(Integer)
    date_exam      = Column(DateTime)
    date_insertion = Column(DateTime)

    under_penetrated                = Column(Boolean)
    over_penetrated                 = Column(Boolean)
    costophrenic_cropped            = Column(Boolean)
    apices_cropped                  = Column(Boolean)
    reliable_radiography            = Column(Boolean)
    minimum_interpretation_quality  = Column(Boolean)
    performed_by                    = Column(String)
    comment                         = Column(String)
    metadata                        = Column(String, default="{}")


class User(Base):

    __tablename__ = "user"
    id             = Column(Integer, primary_key = True)
    userid         = Column(String, unique=True)
    username       = Column(String)
    full_name      = Column(String)
    email          = Column(String)
    hashed_password= Column(String)
    disabled       = Column(Boolean, default=False)
    level          = Column(String)
