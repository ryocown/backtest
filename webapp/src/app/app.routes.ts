import { Routes } from '@angular/router';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { ConfigurationComponent } from './components/configuration/configuration.component';

export const routes: Routes = [
  { path: '', component: DashboardComponent },
  { path: 'config', component: ConfigurationComponent },
  { path: '**', redirectTo: '' }
];
