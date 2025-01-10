__all__ = ["Dataset", "Image", "ImageInfo", "User"]


import datetime, traceback, os

from sqlalchemy import create_engine, Column, Boolean, Float, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base





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
  
