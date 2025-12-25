import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject, map, tap } from 'rxjs';
import { DataPackage } from '../models/data-models';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  private dataSubject = new BehaviorSubject<DataPackage | null>(null);
  public data$ = this.dataSubject.asObservable();

  constructor(private http: HttpClient) { }

  loadLatestData(filename: string = 'latest_backtest.json'): Observable<DataPackage> {
    return this.http.get<DataPackage>(`data/${filename}`).pipe(
      tap(data => this.dataSubject.next(data)),
      tap(data => console.log('Loaded data package:', data))
    );
  }

  getBacktestData() {
    return this.data$.pipe(map(data => data?.backtest || null));
  }

  getTdaData() {
    return this.data$.pipe(map(data => data?.tda || null));
  }
}
